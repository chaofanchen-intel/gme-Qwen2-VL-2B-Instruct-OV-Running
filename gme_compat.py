"""transformers-version compatibility for the 01..05 pipeline scripts.

The pipeline was originally written against transformers 4.x. transformers 5.0 renamed or
moved several names it depends on, and one of them breaks code we do not own: the GME
checkpoint ships its own ``gme_inference.py`` (used by 01_reference.py and
04_validate_python.py for the PyTorch golden, and by 02_export.py for ``fetch_image``) whose
first lines are ``from transformers import AutoModelForVision2Seq, AutoProcessor``.

Rather than editing a vendored upstream file, alias the old name back into the
``transformers`` namespace. Call :func:`patch_transformers` before importing
``gme_inference``.

The other v5 moves are handled at their use sites, since they need per-object access rather
than a module-level alias:

* ``base.model.embed_tokens``      -> ``base.model.get_input_embeddings()``
* ``base.visual``                  -> ``base.model.visual``
* ``base.config.hidden_size``      -> ``base.config.get_text_config().hidden_size``

See docs/09_round3_further_latency_optimization.md S9.3.
"""
from __future__ import annotations

_patched = False


def patch_transformers() -> None:
    """Idempotently restore transformers 4.x names that 5.x removed."""
    global _patched
    if _patched:
        return
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        from transformers import AutoModelForImageTextToText

        transformers.AutoModelForVision2Seq = AutoModelForImageTextToText
    _patched = True


def auto_vlm_class():
    """The auto class for a vision-language model, under either transformers major."""
    patch_transformers()
    import transformers

    return transformers.AutoModelForVision2Seq
