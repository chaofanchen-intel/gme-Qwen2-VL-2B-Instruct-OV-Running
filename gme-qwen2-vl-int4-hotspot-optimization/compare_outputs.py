#!/usr/bin/env python3
"""Compare two dump_outputs.py .npz files (cosine similarity + max abs diff)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ref, new = np.load(sys.argv[1]), np.load(sys.argv[2])
    worst_cos, worst_key = 1.0, ""
    print(f"{'case':16s} {'cosine':>12s} {'max_abs_diff':>14s} {'rel_l2':>10s}")
    for key in ref.files:
        a, b = ref[key].astype(np.float64), new[key].astype(np.float64)
        assert a.shape == b.shape, f"{key}: {a.shape} vs {b.shape}"
        cos = float(np.min([
            np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            for x, y in zip(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
        ]))
        mad = float(np.max(np.abs(a - b)))
        rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
        print(f"{key:16s} {cos:12.9f} {mad:14.3e} {rel:10.3e}")
        if cos < worst_cos:
            worst_cos, worst_key = cos, key
    print(f"\nworst cosine: {worst_cos:.9f} ({worst_key})")
    sys.exit(0 if worst_cos > 0.99999 else 1)


if __name__ == "__main__":
    main()
