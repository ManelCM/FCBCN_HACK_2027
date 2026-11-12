"""E119 (ADR-0023, zona gris): ¿se puede fijar `dead_weight` del banco sin la GT pública?

En E006, `dead_weight` (cuánto más probable es abrir un hueco de balón donde el limpio no tenía balón,
~muerto) se calibró con varios objetivos. Uno era el «56 % de D en los huecos largos» de la GT pública:
con la aclaración del 3 nov eso es calibrar con la GT. Aquí se usa otro objetivo que **solo sale del
tracking del reto** (E006, `atd_stats.long_gap_report`, sin GT): la velocidad colectiva mediana de los
jugadores en los huecos largos de balón (0,97 m/s) frente a la de los frames con balón (1,62 m/s).
Si el balón se pierde sobre todo en parones, en los huecos los jugadores van más despacio.

Para cada `dead_weight` se degrada solo el balón de los 10 partidos del banco (como `calibrate_ball`,
misma semilla) y se mide lo mismo que en el ATD. Solo datos externos; el objetivo son dos agregados.

    uv run python -m hackbarca.pipeline.dead_weight_check
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hackbarca.io.degrade import _long_ball
from hackbarca.pipeline.atd_stats import LONG_GAP, collective_speed
from hackbarca.pipeline.bench_config import BENCH_DEGRADE_V3_1
from hackbarca.pipeline.calibrate_ball import bench_inputs

ATD_GAP_SPEED, ATD_BALL_SPEED = 0.97, 1.62     # E006, solo tracking del reto
GRID = (10.0, 30.0, 100.0, 300.0, 1000.0)


def long_gaps(bx: np.ndarray) -> np.ndarray:
    missing = np.isnan(bx)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    out = np.zeros(len(bx), dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a > LONG_GAP and a > 0 and b < len(bx):
            out[a:b] = True
    return out


def main() -> int:
    inputs = bench_inputs()
    speeds = [collective_speed(m) for m, *_ in inputs]
    print("objetivo ATD: hueco largo %.2f  con balón %.2f  cociente %.3f" % (
        ATD_GAP_SPEED, ATD_BALL_SPEED, ATD_GAP_SPEED / ATD_BALL_SPEED))
    print("dead_weight | v hueco | v balón | cociente | ausente | cubre huecos largos | D en huecos largos (GT banco)")
    for dw in GRID:
        params = replace(BENCH_DEGRADE_V3_1, dead_weight=dw)
        gap_v, ball_v, missing, cover, dead_share = [], [], [], [], []
        for i, ((match, visible, weight, graded, live), speed) in enumerate(zip(inputs, speeds)):
            rng = np.random.default_rng(i)
            bx, _ = _long_ball(rng, match, params, visible, weight)
            gaps = long_gaps(bx) & graded
            seen = ~np.isnan(bx) & graded
            gap_v.append(speed[gaps])
            ball_v.append(speed[seen])
            missing.append(np.isnan(bx[graded]))
            cover.append(gaps[graded])
            dead_share.append(~live[gaps])
        g, b = np.nanmedian(np.concatenate(gap_v)), np.nanmedian(np.concatenate(ball_v))
        mark = "  <- banco v3.1" if dw == BENCH_DEGRADE_V3_1.dead_weight else ""
        print("%11.0f | %7.2f | %7.2f | %8.3f | %7.3f | %19.3f | %.3f%s" % (
            dw, g, b, g / b, np.concatenate(missing).mean(), np.concatenate(cover).mean(),
            np.concatenate(dead_share).mean(), mark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
