"""U3 (E092): toques desde el balon del video sin filtrar frente a los del ATD. Solo agregados (ADR-0004).

    scripts/run_aggregate.sh video_touch uv run python -m hackbarca.pipeline.video_touch_report

El ATD aplana las patadas (E013); la pasada de video (E074) tiene el balon a 25 fps reales. Toque = maximo
local de |v(t+2) - v(t-2)| (cambio de velocidad en 0,16 s) por encima de JUMP m/s, a >= 0,3 s del anterior,
sobre la pista del candidato mas fiable por frame (sin saltos de mas de 2 m entre frames).

Puerta (pseudoetiquetas de la entrega, ADR-0019): en los cambios H<->A directos predichos, fraccion con un
toque a <= 0,5 s, frente a la misma fraccion en instantes al azar de juego vivo predicho (la base). Pasa si
el video tiene un lift >= 1,5 sobre su base Y cubre cambios que el ATD no ve (>= 10 % de los cambios con
toque del video y sin toque del ATD).
"""

from __future__ import annotations

import glob
import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.pipeline.atd_stats import GRADED

FPS = 25
JUMP = 3.0      # m/s de cambio de velocidad en 0,16 s
GAP = 8         # frames entre toques
NEAR = 12       # frames: 0,5 s


def _data(*parts):
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), *parts)


def video_ball(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, y, conf = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    for f in sorted(glob.glob(_data("derived", "video_pass_full", "block*.npz"))):
        for fr, bx, by, c in np.load(f)["ball"]:
            fr = int(fr)
            if c > conf[fr]:
                x[fr], y[fr], conf[fr] = bx, by, c
    step = np.hypot(np.diff(x), np.diff(y))
    bad = np.concatenate(([False], step > 2.0))
    x[bad], y[bad] = np.nan, np.nan
    return x, y


def touches(x: np.ndarray, y: np.ndarray, jump: float = JUMP) -> np.ndarray:
    vx, vy = np.gradient(x) * FPS, np.gradient(y) * FPS
    dv = np.full(len(x), np.nan)
    dv[2:-2] = np.hypot(vx[4:] - vx[:-4], vy[4:] - vy[:-4])
    dv = np.nan_to_num(dv)
    cand = np.flatnonzero((dv >= jump) & (dv >= np.roll(dv, 1)) & (dv >= np.roll(dv, -1)))
    out, last = [], -GAP
    for t in cand:
        if t - last >= GAP:
            out.append(t)
            last = t
    return np.array(out, dtype=int)


def near(events: np.ndarray, t: np.ndarray) -> np.ndarray:
    if not len(events):
        return np.zeros(len(t), dtype=bool)
    k = np.clip(np.searchsorted(events, t), 1, len(events) - 1)
    return np.minimum(np.abs(events[k - 1] - t), np.abs(events[k] - t)) <= NEAR


def bench_rate(jump: float = JUMP) -> float:
    """Toques por minuto de juego vivo con el balon limpio del banco v3 (el ritmo real de toques)."""
    from hackbarca.io.external import load_bench_game
    from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
    hits, minutes = 0, 0.0
    for name in BENCH_GAMES_V3:
        match, gt = load_bench_game(name)
        live = np.isin(gt, (0, 1))
        hits += live[touches(match.ball_x, match.ball_y, jump)].sum()
        minutes += live.sum() / FPS / 60
    return hits / minutes


def main() -> int:
    from hackbarca.io.load_match import load_match
    match = load_match()
    n = match.n_frames
    graded = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    vx, vy = video_ball(n)
    target = bench_rate()
    # umbral del video: el que da en juego vivo predicho el ritmo de toques del balon limpio del banco
    live = graded & np.isin(pred, (0, 1))
    live_min = live.sum() / FPS / 60
    rates = {j: live[touches(vx, vy, j)].sum() / live_min for j in (3, 5, 8, 12, 16, 20, 25, 30)}
    jump = min(rates, key=lambda j: abs(rates[j] - target))
    print("toques por minuto de juego vivo: banco limpio %.1f | video por umbral %s -> umbral %g" % (
        target, " ".join("%g:%.1f" % kv for kv in rates.items()), jump))
    tv, ta = touches(vx, vy, jump), touches(match.ball_x, match.ball_y)
    minutes = graded.sum() / FPS / 60
    live = graded & np.isin(pred, (0, 1))
    print("cobertura del balon: video %.3f, ATD %.3f de los frames de juego vivo predicho" % (
        np.isfinite(vx[live]).mean(), np.isfinite(match.ball_x[live]).mean()))
    print("toques por minuto: video %.1f, ATD %.1f" % (graded[tv].sum() / minutes, graded[ta].sum() / minutes))
    segs = [s for r in GRADED for s in pm.segments(pred, [tuple(r)])]
    chg = np.array([b0 for (a0, b0, c0), (a1, b1, c1) in zip(segs[:-1], segs[1:]) if {c0, c1} == {0, 1} and b0 == a1])
    rng = np.random.default_rng(0)
    base_t = rng.choice(np.flatnonzero(live), 20000, replace=False)
    for name, ev in (("video", tv), ("ATD", ta), ("video o ATD", np.union1d(tv, ta))):
        hit, base = near(ev, chg).mean(), near(ev, base_t).mean()
        print("%-12s cambios con toque a <= 0,5 s %.3f | base %.3f | lift %.2f" % (name, hit, base, hit / max(base, 1e-9)))
    only_video = near(tv, chg) & ~near(ta, chg)
    print("cambios con toque del video y sin toque del ATD: %.3f (n cambios %d)" % (only_video.mean(), len(chg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
