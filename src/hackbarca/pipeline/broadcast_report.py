"""Puerta de decision del Frente A: cuanto balon aporta el broadcast alineado (solo agregados).

ADR-0004: lee data/ (tactico y broadcast), lo ejecuta una persona:

    uv run python -m hackbarca.pipeline.broadcast_report

Guarda la correspondencia tactico -> broadcast en data/derived/broadcast_alignment.npz (dentro de
data/, ignorado por git) para que la fusion no tenga que recalcularla.
"""

from __future__ import annotations

import os
import time

import numpy as np

from hackbarca.io.broadcast_align import align, alignment_report
from hackbarca.io.load_match import load_match
from hackbarca.pipeline.atd_stats import GRADED


def main() -> int:
    started = time.time()
    tactical = load_match(source="tactical")
    broadcast = load_match(source="broadcast")
    print("cargado en {:.0f}s: tactico {} frames, broadcast {} frames".format(
        time.time() - started, tactical.n_frames, broadcast.n_frames))

    started = time.time()
    result = align(tactical, broadcast)
    print("alineado en {:.0f}s".format(time.time() - started))

    graded = np.zeros(tactical.n_frames, dtype=bool)
    for start, stop in GRADED:
        graded[start:stop] = True
    for key, value in alignment_report(tactical, broadcast, result, graded).items():
        print("{:<28} {}".format(key, round(value, 3) if isinstance(value, float) else value))

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    out = os.path.join(data_dir, "derived", "broadcast_alignment.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, tactical_to_broadcast=result.tactical_to_broadcast,
                        flip=np.array(result.flip))
    print("guardado", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
