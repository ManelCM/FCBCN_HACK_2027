"""Agregados del balon relleno por video en paron y en juego vivo: reto frente a banco simulado (E068).

Reto: frames de hueco del ATD rellenados, separados por la prediccion del candidato sin relleno
(submissions/candidato_familias.csv: D frente a H/A; solo como aproximacion, no hay GT). Banco: relleno
simulado separado por la GT. Solo imprime cuantiles (ADR-0004).

    scripts/run_aggregate.sh relleno_parones uv run python -m hackbarca.vision.fill_dead_live
"""
import numpy as np
import pandas as pd
from hackbarca.io.load_match import load_match
from hackbarca.io.video_fill import apply_video_fill, simulate_video_fill, VideoFill
from hackbarca.io.degrade import degrade
from hackbarca.io.external import load_bench_game
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES_V3
from hackbarca.pipeline.atd_stats import GRADED

def stats(m, before_nan, mask):
    x, y = m.ball_x, m.ball_y
    filled = before_nan & ~np.isnan(x) & mask
    sp = np.full(len(x), np.nan)
    sp[1:] = np.hypot(np.diff(x), np.diff(y)) * 25
    d = np.nanmin(np.hypot(m.players_x - x[:, None], m.players_y - y[:, None]), axis=1)
    f = np.flatnonzero(filled)
    f = f[f > 0]
    return len(f), np.nanpercentile(sp[f], [25, 50, 75]).round(2).tolist(), np.nanpercentile(d[f], [25, 50, 75]).round(2).tolist(), float(np.mean(sp[f] < 0.5))

m = load_match()
pred = pd.read_csv("submissions/candidato_familias.csv").iloc[:, -1].astype(str).to_numpy()
graded = np.zeros(m.n_frames, bool)
for a, b in GRADED:
    graded[a:b] = True
gap = np.isnan(m.ball_x)
mf = apply_video_fill(m, "data/derived/video_ball.npz")
for name, sel in (("parón (pred D)", pred == "D"), ("vivo (pred H/A)", np.isin(pred, ["H", "A"]))):
    n, sp, d, still = stats(mf, gap, sel & graded)
    print("RETO  %-16s frames rellenos %6d | velocidad p25/50/75 %s m/s | dist. jugador más cercano p25/50/75 %s m | quieto %.2f" % (name, n, sp, d, still))

acc = {"D": [], "V": []}
for i, g in enumerate(BENCH_GAMES_V3):
    clean, gt = load_bench_game(g)
    deg = degrade(clean, BENCH_DEGRADE, seed=i)
    sim = simulate_video_fill(deg, clean, VideoFill(), seed=i)
    gap_b = np.isnan(deg.ball_x)
    for k, sel in (("D", gt == 2), ("V", np.isin(gt, [0, 1]))):
        x, y = sim.ball_x, sim.ball_y
        filled = gap_b & ~np.isnan(x) & sel
        sp = np.full(len(x), np.nan); sp[1:] = np.hypot(np.diff(x), np.diff(y)) * 25
        dd = np.nanmin(np.hypot(sim.players_x - x[:, None], sim.players_y - y[:, None]), axis=1)
        f = np.flatnonzero(filled); f = f[f > 0]
        acc[k].append(np.column_stack([sp[f], dd[f]]))
for k, name in (("D", "parón (GT D)"), ("V", "vivo (GT H/A)")):
    a = np.concatenate(acc[k])
    print("BANCO %-16s frames rellenos %6d | velocidad p25/50/75 %s m/s | dist. jugador más cercano p25/50/75 %s m | quieto %.2f" % (
        name, len(a), np.nanpercentile(a[:, 0], [25, 50, 75]).round(2).tolist(), np.nanpercentile(a[:, 1], [25, 50, 75]).round(2).tolist(), float(np.mean(a[:, 0] < 0.5))))
