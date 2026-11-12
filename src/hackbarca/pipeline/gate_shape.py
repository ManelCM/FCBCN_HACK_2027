"""Puerta C4: ¿separan la compacidad y la forma quien tiene el balon, y como de rapido tras un robo?

    uv run python -m hackbarca.pipeline.gate_shape

Solo banco (datos externos). Con las features ya en cache del modelo de la entrega, mide el AUC de
"posesion local frente a visitante" en juego vivo de cada senal de forma en version diferencial
(local menos visitante), que es como la describe la literatura (con balon los equipos son mas
anchos, largos y extensos: AMISCO, LJMU). Se mide en tres contextos:
  - todo el juego vivo;
  - dentro de los huecos largos sin balon (donde la forma es casi lo unico que queda);
  - en los 0-3 s tras un cambio H<->A de la GT (lo que fallamos: robos no vistos).
Un AUC de 0,5 es azar; 1,0 separa siempre. Se compara con el AUC de la probabilidad del modelo y
con la senal del jugador mas cercano.
"""

from __future__ import annotations

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.io.degrade import degrade
from hackbarca.io.external import controlled_possession, is_dfl, load_bench_game
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES_V3
from hackbarca.pipeline.tune_l3 import bench_feature_paths, read_features

H, A, D = 0, 1, 2
FEATURES = "touch+ball+players+blocks+structure+restart"
OOF = "oof_l3_notags_v3_ftouch+ball+players+blocks+structure+restart_stack_dyn_ctl1.5_dfl2_trainraw.npz"
LONG_GAP = 100
AFTER = 75
PAIRS = ("blk_length", "blk_width", "blk_area", "blk_stretch", "blk_back", "blk_front", "blk_lines_gap", "blk_lpw",
         "blk_length_chg25", "blk_width_chg25", "blk_area_chg25", "blk_area_chg75", "blk_width_chg75",
         "shape_open", "shape_open_chg25", "shape_open_chg75", "behind_ball", "in_opp_half",
         "in_opp_half_chg25", "in_opp_half_chg75")


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """AUC por rangos (Mann-Whitney), ignorando NaN; y = 1 si posee el local."""
    ok = ~np.isnan(score)
    s, t = score[ok], y[ok]
    n1, n0 = float(t.sum()), float((~t).sum())
    if not n1 or not n0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # empates: rango medio
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    import os
    blob = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "..", "external_data", OOF))
    cols = {"live": {}, "gap": {}, "after": {}}
    store = {k: {} for k in cols}
    for i, (name, (path, _)) in enumerate(zip(BENCH_GAMES_V3, bench_feature_paths(use_tags=False, version=FEATURES,
                                                                                  names=BENCH_GAMES_V3))):
        match, raw = load_bench_game(name)
        gt = controlled_possession(raw, int(round((2.0 if is_dfl(name) else 1.5) * match.fps)))
        df = read_features(path)
        known = ~np.isnan(degrade(match, BENCH_DEGRADE, seed=i).ball_x)
        edges = np.diff(np.concatenate(([0], (~known).astype(np.int8), [0])))
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
        masks = {"live": live, "gap": live & gap, "after": live & after}
        signals = {"modelo P(H)-P(A)": blob["p%d" % i][:, 0] - blob["p%d" % i][:, 1]}
        for base in PAIRS:
            if base + "_h" in df and base + "_a" in df:
                signals[base + " (h-a)"] = (df[base + "_h"] - df[base + "_a"]).to_numpy()
        for key in [c for c in df.columns if c.startswith("near") or c.startswith("dist")][:4]:
            signals[key] = df[key].to_numpy()
        for ctx, m in masks.items():
            for sig, values in signals.items():
                store[ctx].setdefault(sig, ([], []))
                store[ctx][sig][0].append(values[m])
                store[ctx][sig][1].append(gt[m] == H)
    print("AUC de 'posee el local' (0,5 = azar; < 0,5 = senal invertida, vale igual: |AUC-0,5|)")
    print("{:<34}{:>10}{:>10}{:>14}".format("senal", "vivo", "en hueco", "0-3 s tras robo"))
    sigs = list(store["live"])
    rows = []
    for sig in sigs:
        vals = [auc(np.concatenate(store[c][sig][0]), np.concatenate(store[c][sig][1])) for c in ("live", "gap", "after")]
        rows.append((abs(vals[0] - 0.5), sig, vals))
    for _, sig, vals in sorted(rows, reverse=True):
        print("{:<34}{:>10.3f}{:>10.3f}{:>14.3f}".format(sig, *vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
