import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from mmdocir_eval.data import Document, RowRange
from mmdocir_eval.runner import EncodeControls, cleanup_stale_temporaries, encode_layouts

LAYOUT_COUNT = 8
# Every fourth row is an image, so the text and fused queues drain out of step
# and the resume watermark has to be a completed *prefix* rather than a count.
IMAGE_POSITIONS = {2, 5}


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_layouts_parquet(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    image = _png_bytes()
    table = pa.table(
        {
            "doc_name": ["d.pdf"] * LAYOUT_COUNT,
            "type": ["image" if i in IMAGE_POSITIONS else "text" for i in range(LAYOUT_COUNT)],
            "layout_id": list(range(LAYOUT_COUNT)),
            "page_id": [0] * LAYOUT_COUNT,
            "bbox": [[0.0, 0.0, 1.0, 1.0]] * LAYOUT_COUNT,
            "image_binary": [image if i in IMAGE_POSITIONS else b"" for i in range(LAYOUT_COUNT)],
            "text": [f"t{i}" for i in range(LAYOUT_COUNT)],
            "vlm_text": [f"t{i}" for i in range(LAYOUT_COUNT)],
        }
    )
    parquet.write_table(table, path)


def _documents() -> list[Document]:
    return [
        Document(
            doc_name="d.pdf",
            domain="test",
            page_rows=RowRange(0, 0),
            layout_rows=RowRange(0, LAYOUT_COUNT - 1),
            queries=(),
        )
    ]


class StubBackend:
    """Encodes "t<n>" into the vector [n, 1] so every row is checkable by position."""

    model_name = "stub"
    embedding_dim = 2

    def __init__(self, fail_after: int | None = None) -> None:
        self.encoded: list[str] = []
        self.fail_after = fail_after

    def _encode(self, texts):
        if self.fail_after is not None and len(self.encoded) >= self.fail_after:
            raise RuntimeError("simulated wedge")
        self.encoded.extend(texts)
        return np.asarray([[float(text[1:]), 1.0] for text in texts], dtype=np.float32)

    def encode_queries(self, texts, batch_size=None):
        return self._encode(texts)

    def encode_page_images(self, images, batch_size=None):
        raise AssertionError("pages are not exercised here")

    def encode_layout_texts(self, texts, batch_size=None):
        return self._encode(texts)

    def encode_layout_fused(self, texts, images, batch_size=None):
        return self._encode(texts)

    def metadata(self):
        return {"adapter": "stub"}


def _expected_vectors() -> np.ndarray:
    raw = np.asarray([[float(i), 1.0] for i in range(LAYOUT_COUNT)], dtype=np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


class ResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.parquet = self.root / "layouts.parquet"
        _write_layouts_parquet(self.parquet)
        self.documents = _documents()
        self.cache_key = {"backend": "stub"}

    def _encode(self, backend, *, resume: bool):
        controls = EncodeControls(
            label="[stub]",
            progress_interval_seconds=0.0,
            checkpoint_interval_seconds=0.0,
            resume=resume,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            index = encode_layouts(
                backend,
                self.parquet,
                self.documents,
                self.cache,
                self.cache_key,
                text_batch_size=2,
                fused_batch_size=1,
                force=False,
                controls=controls,
            )
        return index, output.getvalue()

    def _partial_paths(self) -> tuple[Path, Path]:
        temporaries = sorted(self.cache.glob("*.npy.tmp"))
        progresses = sorted(self.cache.glob("*.progress.json"))
        self.assertEqual(len(temporaries), 1, temporaries)
        self.assertEqual(len(progresses), 1, progresses)
        return temporaries[0], progresses[0]

    def test_resume_encodes_only_the_missing_rows(self) -> None:
        crashed = StubBackend(fail_after=4)
        with self.assertRaises(RuntimeError):
            self._encode(crashed, resume=True)
        _, progress_path = self._partial_paths()
        checkpoint = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = checkpoint["completed"]
        self.assertGreater(completed, 0)
        self.assertLess(completed, LAYOUT_COUNT)

        resumed = StubBackend()
        index, log = self._encode(resumed, resume=True)
        self.assertIn(f"resuming layout at {completed}/{LAYOUT_COUNT}", log)
        # The rows the first attempt already wrote are not paid for twice.
        self.assertEqual(resumed.encoded, [f"t{i}" for i in range(completed, LAYOUT_COUNT)])
        np.testing.assert_allclose(np.asarray(index.vectors), _expected_vectors(), rtol=1e-6)
        self.assertEqual(index.performance["resumed_from"], completed)
        self.assertEqual([row["layout_id"] for row in index.rows], list(range(LAYOUT_COUNT)))
        self.assertEqual(sorted(self.cache.glob("*.npy.tmp")), [])
        self.assertEqual(sorted(self.cache.glob("*.progress.json")), [])

    def test_without_resume_the_partial_cache_is_rebuilt(self) -> None:
        with self.assertRaises(RuntimeError):
            self._encode(StubBackend(fail_after=4), resume=True)
        self._partial_paths()

        rebuilt = StubBackend()
        index, _ = self._encode(rebuilt, resume=False)
        self.assertEqual(rebuilt.encoded, [f"t{i}" for i in range(LAYOUT_COUNT)])
        self.assertEqual(index.performance["resumed_from"], 0)
        np.testing.assert_allclose(np.asarray(index.vectors), _expected_vectors(), rtol=1e-6)

    def test_checkpoint_from_another_configuration_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            self._encode(StubBackend(fail_after=4), resume=True)
        _, progress_path = self._partial_paths()
        checkpoint = json.loads(progress_path.read_text(encoding="utf-8"))
        checkpoint["fingerprint"] = "0" * 16
        progress_path.write_text(json.dumps(checkpoint), encoding="utf-8")

        backend = StubBackend()
        index, log = self._encode(backend, resume=True)
        self.assertIn("fingerprint differs", log)
        self.assertEqual(backend.encoded, [f"t{i}" for i in range(LAYOUT_COUNT)])
        np.testing.assert_allclose(np.asarray(index.vectors), _expected_vectors(), rtol=1e-6)

    def test_completed_cache_is_reused_without_touching_the_backend(self) -> None:
        self._encode(StubBackend(), resume=False)
        index, _ = self._encode(StubBackend(fail_after=0), resume=False)
        self.assertTrue(index.performance["cache_hit"])


class CleanupTest(unittest.TestCase):
    def test_removes_only_temporaries_without_a_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            orphan = cache / "layout_dead.npy.tmp"
            orphan.write_bytes(b"")
            keeper = cache / "layout_live.npy.tmp"
            keeper.write_bytes(b"")
            (cache / "layout_live.progress.json").write_text("{}", encoding="utf-8")

            removed, resumable = cleanup_stale_temporaries(cache)

            self.assertEqual(removed, [orphan])
            self.assertEqual(resumable, [keeper])
            self.assertFalse(orphan.exists())
            self.assertTrue(keeper.exists())


if __name__ == "__main__":
    unittest.main()
