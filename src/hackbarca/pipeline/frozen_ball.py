"""¿Tiene el ATD real tramos de balon congelado o lento en juego vivo que el banco no simula? (E043)

    scripts/run_aggregate.sh frozen_ball uv run python -m hackbarca.pipeline.frozen_ball

short_dead_report (E042) atribuye los D cortos del reto sobre todo a `ball_speed` y `touch_rank`:
dentro de ellos el balon del ATD va tan lento como en un paron real. Aqui se mide, con el balon
visible, la velocidad del balon y los tramos "congelados" (posicion repetida o casi: < 0,05 m por
frame) en el reto, separados por lo que predecimos (vivo, D corto, D largo), y en el banco degradado
separados por la GT (vivo, D). Solo cuantiles y tasas.
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.io.degrade import degrade
from hackbarca.io.external import controlled_possession, is_dfl, load_bench_game
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES_V3

H, A, D = 0, 1, 2
FPS = 25
STILL = 0.05     # m por frame (1,25 m/s)
FROZEN = 0.001   # m por frame: posicion repetida


def ball_step(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    step = np.full(len(x), np.nan)
    step[1:] = np.hypot(np.diff(x), np.diff(y))
    return step


def run_lengths(mask: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def describe(label: str, step: np.ndarray, sel: np.ndarray) -> dict:
    s = step[sel & np.isfinite(step)]
    if not len(s):
        return {}
    still = sel & np.isfinite(step) & (step < STILL)
    frozen = sel & np.isfinite(step) & (step < FROZEN)
    minutes = sel.sum() / FPS / 60
    long_still = run_lengths(still)
    row = dict(frames=int(sel.sum()), speed_p25=float(np.percentile(s, 25) * FPS),
               speed_p50=float(np.percentile(s, 50) * FPS), still_share=float(np.mean(s < STILL)),
               frozen_share=float(np.mean(s < FROZEN)),
               still_runs_1s_per_min=float((long_still >= FPS).sum() / max(minutes, 1e-9)))
    print("  {:<34} frames {:7d} | vel. p25/p50 {:5.2f}/{:5.2f} m/s | lento {:5.1%} | congelado {:5.1%} | "
          "tramos lentos >=1 s por minuto {:5.2f}".format(label, row["frames"], row["speed_p25"], row["speed_p50"],
                                                          row["still_share"], row["frozen_share"],
                                                          row["still_runs_1s_per_min"]))
    return row


def main() -> int:
    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED

    print("RETO (balon visible del ATD; grupos por nuestra prediccion de la entrega)")
    match = load_match()
    step = ball_step(match.ball_x, match.ball_y)
    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    graded = np.zeros(match.n_frames, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    short = np.zeros(match.n_frames, dtype=bool)
    long_d = np.zeros(match.n_frames, dtype=bool)
    for a, b in GRADED:
        for s, e, c in pm.segments(pred, [(a, b)]):
            if c == D:
                (short if e - s < 125 else long_d)[s:e] = True
    visible = graded & ~np.isnan(match.ball_x)
    describe("vivo predicho", step, visible & (pred != D))
    describe("D corto predicho (< 5 s)", step, visible & short)
    describe("D largo predicho (>= 5 s)", step, visible & long_d)

    print("\nBANCO degradado (balon visible; grupos por la GT controlada)")
    live_steps, dead_steps = [], []
    for i, name in enumerate(BENCH_GAMES_V3):
        m, raw = load_bench_game(name)
        gt = controlled_possession(raw, int(round((2.0 if is_dfl(name) else 1.5) * m.fps)))
        deg = degrade(m, BENCH_DEGRADE, seed=i)
        st = ball_step(deg.ball_x, deg.ball_y)
        vis = ~np.isnan(deg.ball_x)
        live_steps.append(np.where(vis & ((gt == H) | (gt == A)), st, np.nan))
        dead_steps.append(np.where(vis & (gt == D), st, np.nan))
    for label, parts in (("vivo (GT)", live_steps), ("D (GT)", dead_steps)):
        st = np.concatenate([np.concatenate([p, [np.nan]]) for p in parts])
        describe(label, st, np.isfinite(st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
