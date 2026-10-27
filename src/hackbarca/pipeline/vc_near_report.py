"""Agregado (E075s): acuerdo del conductor por video con el equipo predicho segun la distancia al cambio
predicho mas cercano, en el reto (video real) y en el banco (video simulado, base y estres). Solo cuotas.

    scripts/run_aggregate.sh vc_near uv run python -m hackbarca.pipeline.vc_near_report
"""
import os
import numpy as np
from hackbarca.io.external import load_bench_game
from hackbarca.io.load_match import load_match
from hackbarca.io.video_conductor import (FPS, VIDEO_CONDUCTOR, VIDEO_CONDUCTOR_STRESS, real_conductor,
                                          simulate_conductor)
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.bench_config import BENCH_GAMES_V3

BINS = ((0, 1.5), (1.5, 5), (5, 1e9))


def dist_to_change(pred):
    live = np.flatnonzero(np.isin(pred, (0, 1)))
    chg = live[1:][pred[live[1:]] != pred[live[:-1]]]
    idx = np.arange(len(pred))
    k = np.searchsorted(chg, idx)
    return np.minimum(np.abs(idx - chg[np.maximum(k - 1, 0)]), np.abs(chg[np.minimum(k, len(chg) - 1)] - idx)) / FPS


def stats(video, pred, gt, mask):
    d = dist_to_change(pred)
    grid = np.zeros(len(pred), dtype=bool)
    grid[::5] = True
    out = []
    for a, b in BINS:
        sel = mask & grid & np.isin(pred, (0, 1)) & (d > a) & (d <= b)
        obs = sel & np.isfinite(video)
        out.append((obs.sum(), sel.sum(), (video[obs] == pred[obs]).sum(),
                    np.nan if gt is None else (pred[sel] == gt[sel]).sum()))
    return np.array(out, dtype=float)


def show(name, s, acc=None):
    for (a, b), (no, ns, ag, ok) in zip(BINS, s):
        agree = ag / no
        line = "%-14s %4.1f-%-4s s: acuerdo %.3f cobertura %.3f (n=%d)" % (name, a, "inf" if b > 1e8 else "%.1f" % b,
                                                                       agree, no / ns, no)
        if acc is not None:
            m = acc[BINS.index((a, b))]
            line += " | modelo %.3f -> video implicito %.3f" % (m, (agree - (1 - m)) / (2 * m - 1))
        print(line)


z = np.load(os.path.join("external_data", "oof_l3_notags_v3_1_ftouch+ball+players+blocks+structure+restart_stack_ctl1.5_dfl2_trainraw.npz"))
bench = {"base": 0, "estres": 0}
for i, name in enumerate(BENCH_GAMES_V3):
    match, gt = load_bench_game(name)
    p = z["p%d" % i]
    assert len(p) == match.n_frames
    pred = p.argmax(1)
    g = z["gt%d" % i]
    for label, sim in (("base", VIDEO_CONDUCTOR), ("estres", VIDEO_CONDUCTOR_STRESS)):
        bench[label] = bench[label] + stats(simulate_conductor(match, gt, sim, seed=i), pred, g, np.ones(len(pred), bool))
acc = bench["base"][:, 3] / bench["base"][:, 1]
print("banco: acierto del modelo (OOF E049, argmax) por distancia al cambio predicho:", " / ".join("%.3f" % a for a in acc))
show("banco base", bench["base"], acc)
show("banco estres", bench["estres"], acc)

match = load_match()
graded = np.zeros(match.n_frames, dtype=bool)
for a, b in GRADED:
    graded[a:b] = True
pred = np.load(os.path.join("data", "derived", "proba_l3.npz"))["proba"].argmax(1)
show("reto", stats(real_conductor(match), pred, None, graded), acc)
