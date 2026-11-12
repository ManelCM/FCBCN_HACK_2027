"""Reproducibilidad de la pasada de video (variante A): compara una repeticion con la pasada guardada.

    uv run --group vision python -m hackbarca.vision.video_pass --limit 1000 --out-dir repro_check
    scripts/run_aggregate.sh repro uv run python -m hackbarca.vision.repro_check

Compara los frames de la repeticion (data/derived/repro_check/block000.npz) con los mismos frames de
data/derived/video_pass_full/block000.npz. Solo imprime recuentos, si son identicos y la diferencia
maxima (ADR-0004).
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", default="repro_check")
    args = parser.parse_args(argv)
    root = os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "derived")
    new = np.load(os.path.join(root, args.rerun, "block000.npz"))
    old = np.load(os.path.join(root, "video_pass_full", "block000.npz"))
    for key in ("ball", "people"):
        a = new[key]
        if not len(a):
            print("%s: repeticion vacia" % key)
            continue
        last = a[:, 0].max()
        b = old[key][old[key][:, 0] <= last]
        same_shape = a.shape == b.shape
        print("%s: filas repeticion %d, guardada %d" % (key, len(a), len(b)))
        if same_shape:
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print("  identicos: %s; diferencia maxima %.3g; filas distintas %d" % (
                bool(np.array_equal(a, b, equal_nan=True)), np.nanmax(diff), int((np.nan_to_num(diff) > 0).any(1).sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
