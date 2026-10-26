"""Puerta de M2a (E051): ¿ve la familia "control" los robos que no vemos?

    uv run python -m hackbarca.pipeline.gate_control                    # banco v3.1 (datos externos)
    scripts/run_aggregate.sh control_shift uv run python -m hackbarca.pipeline.gate_control --shift

Banco: calcula `features/control.py` sobre los 10 partidos degradados con v3.1 (misma semilla que
el banco) y mide, como `gate_shape` (E046), el AUC de "posee el local" de cada senal en version
local - visitante: en juego vivo, en huecos largos y en los 0-3 s tras un cambio H<->A de la GT. La
puerta (docs/05_plan_mejora.md, M2a) pide AUC >= 0,80 tras un robo en alguna senal (hoy el jugador
mas cercano da 0,71 y el modelo 0,86) y desajuste reto/banco < 0,25.
--shift: percentiles de cada columna en el reto frente al banco v3.1 (solo agregados, ADR-0004).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hackbarca.eval.evaluate import pm
from hackbarca.features.control import control_features
from hackbarca.io.degrade import degrade
from hackbarca.io.external import controlled_possession, is_dfl, load_bench_game
from hackbarca.models.l1 import interpolate_ball, team_distances
from hackbarca.pipeline.bench_config import BENCH_DEGRADE_V3_1, BENCH_GAMES_V3
from hackbarca.pipeline.gate_shape import AFTER, LONG_GAP, auc

H, A, D = 0, 1, 2


def features_of(match) -> tuple[pd.DataFrame, np.ndarray]:
    bx, by, _ = interpolate_ball(match, 10)          # como frame_features
    known = ~np.isnan(bx)
    f = pd.DataFrame(control_features(match, bx, by, known))
    dh, da = team_distances(match, np.nan_to_num(bx), np.nan_to_num(by))
    f["near_sign"] = np.where(known, np.where(dh <= da, 1.0, -1.0), np.nan)
    return f, known


def bench_games():
    for i, name in enumerate(BENCH_GAMES_V3):
        match, raw = load_bench_game(name)
        gt = controlled_possession(raw, int(round((2.0 if is_dfl(name) else 1.5) * match.fps)))
        deg = degrade(match, BENCH_DEGRADE_V3_1, seed=i)
        f, known = features_of(deg)
        yield name, gt, f, np.isnan(deg.ball_x)


def context_masks(gt, missing):
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    gap = np.zeros(len(gt), dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a > LONG_GAP:
            gap[a:b] = True
    after = np.zeros(len(gt), dtype=bool)
    for s, e in pm.valid_runs(gt):
        w = gt[s:e]
        for t in np.flatnonzero(w[1:] != w[:-1]) + 1:
            if {int(w[t - 1]), int(w[t])} == {H, A}:
                after[s + t:s + t + AFTER] = True
    live = (gt == H) | (gt == A)
    return {"vivo": live, "en hueco": live & gap, "0-3 s tras robo": live & after}


def signals(f: pd.DataFrame) -> dict:
    out = {"jugador mas cercano": f["near_sign"].to_numpy()}
    for col in f.columns:
        if col.endswith("_h") and col[:-2] + "_a" in f:
            out[col[:-2] + " (h-a)"] = (f[col] - f[col[:-2] + "_a"]).to_numpy()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", action="store_true", help="desajuste reto/banco (datos reales, agregados)")
    args = parser.parse_args(argv)
    if args.shift:
        from hackbarca.io.load_match import load_match
        from hackbarca.pipeline.atd_stats import GRADED
        from hackbarca.pipeline.feature_shift import shift_table
        real, _ = features_of(load_match())
        graded = np.zeros(len(real), dtype=bool)
        for a, b in GRADED:
            graded[a:b] = True
        bench = pd.concat([f[gt != 3] for _, gt, f, _ in bench_games()], ignore_index=True)
        rows = shift_table(real[graded], bench, list(real.columns))
        print("{:<22} {:>24} {:>24} {:>8}".format("feature", "reto p10/p50/p90", "banco p10/p50/p90", "desaj."))
        for col, pr, pb, gap in sorted(rows, key=lambda r: -r[3]):
            print("{:<22} {:>24} {:>24} {:>8.2f}".format(
                col, "/".join("%.2f" % v for v in pr), "/".join("%.2f" % v for v in pb), gap))
        return 0

    store = {}
    for name, gt, f, missing in bench_games():
        masks = context_masks(gt, missing)
        for sig, values in signals(f).items():
            for ctx, m in masks.items():
                store.setdefault(sig, {}).setdefault(ctx, ([], []))
                store[sig][ctx][0].append(values[m])
                store[sig][ctx][1].append(gt[m] == H)
        print("listo", name, flush=True)
    print("\nAUC de 'posee el local' (|AUC-0,5| es la fuerza; < 0,5 = senal invertida)")
    print("{:<26}{:>10}{:>10}{:>16}".format("senal", "vivo", "en hueco", "0-3 s tras robo"))
    rows = []
    for sig, ctxs in store.items():
        vals = [auc(np.concatenate(ctxs[c][0]), np.concatenate(ctxs[c][1])) for c in ("vivo", "en hueco", "0-3 s tras robo")]
        rows.append((abs(vals[2] - 0.5), sig, vals))
    for _, sig, vals in sorted(rows, reverse=True):
        print("{:<26}{:>10.3f}{:>10.3f}{:>16.3f}".format(sig, *vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
