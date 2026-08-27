#!/usr/bin/env python3
"""Download only the three files required for MMDocIR retrieval evaluation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPOSITORY = "MMDocIR/MMDocIR_Evaluation_Dataset"
FILES = ("MMDocIR_annotations.jsonl", "MMDocIR_pages.parquet", "MMDocIR_layouts.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/chaofan/datasets/MMDocIR_Evaluation_Dataset"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPOSITORY, files_metadata=True)
    snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        revision=info.sha,
        allow_patterns=list(FILES),
        local_dir=args.output_dir,
        max_workers=3,
    )
    sibling_by_name = {sibling.rfilename: sibling for sibling in info.siblings}
    manifest = {
        "repository": REPOSITORY,
        "revision": info.sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "expected_bytes": sibling_by_name[name].size,
            }
            for name in FILES
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()