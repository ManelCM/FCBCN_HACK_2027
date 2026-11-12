"""E120 (I6 de otros equipos): mover cada frontera predicha al toque mas cercano del equipo que empieza.

Sin entrenar. Sobre un OOF del banco v3.1, con su decodificacion y la regla de D cortos de 3 s: para cada
tramo predicho H o A que empieza en b, se busca en [b - w, b + w] un toque (features `since_touch_log` = 0)
cuyo `last_touch_sign` sea el del equipo del tramo, y la frontera se lleva al mas cercano. Nunca se borra un
tramo: el nuevo inicio queda dentro de los tramos vecinos. Se prueban w = 0,5 / 1 / 2 s y dos variantes:
solo cambios directos H<->A, o tambien las reanudaciones (D -> H/A).

El techo de E040 (fronteras perfectas a <= 1 s) es +0,015; con el balon del banco hay ~0,04 toques/s.

    uv run python -m hackbarca.pipeline.touch_snap --oof <OOF en external_data/> [--decode T,cambio,aD,desdeD]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hackbarca.eval.bench import pm, pooled
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
from hackbarca.pipeline.failure_analysis import drop_short_dead
from hackbarca.pipeline.tune_l3 import BENCH_PARAMS, bench_feature_paths, feature_spec, read_features

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, A, D = 0, 1, 2


def touches(df, home_sign: float):
    """(frames con toque, equipo 0/1 del toque) segun las features de toque."""
    at = np.flatnonzero(np.isclose(df["since_touch_log"].to_numpy(), 0.0))
    sign = df["last_touch_sign"].to_numpy()[at]
    ok = sign != 0
    return at[ok], np.where(sign[ok] == home_sign, H, A)


def snap(pred: np.ndarray, runs, t_frames, t_team, w: int, restarts: bool) -> np.ndarray:
    out = pred.copy()
    for s, e in runs:
        segs = pm.segments(out, [(s, e)])
        for n in range(1, len(segs)):
            a, b, code = segs[n]
            pa, pb, pcode = segs[n - 1]
            if code not in (H, A) or not (pcode in (H, A) and pcode != code or restarts and pcode == D):
                continue
            lo, hi = max(pa + 1, a - w), min(b - 1, a + w)
            sel = (t_frames >= lo) & (t_frames <= hi) & (t_team == code)
            if not sel.any():
                continue
            cand = t_frames[sel]
            new = int(cand[np.argmin(np.abs(cand - a))])
            if new < a:
                out[new:a] = code
            elif new > a:
                out[a:new] = pcode
            segs[n] = (new, b, code)
            segs[n - 1] = (pa, new, pcode)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--decode", default="1,20,10,10", help="T,cambio,a D,desde D")
    parser.add_argument("--features", type=feature_spec, default="touch+ball+players+blocks+structure+restart")
    parser.add_argument("--short-dead", type=int, default=75)
    args = parser.parse_args(argv)
    t, sw, td, fd = (float(x) for x in args.decode.split(","))
    params = DecodeParams(t, sw, td, fd)
    blob = np.load(os.path.join(_REPO_ROOT, "external_data", args.oof))
    paths = bench_feature_paths(use_tags=False, version=args.features, params=BENCH_PARAMS["v3_1"],
                                names=BENCH_GAMES_V3)
    gts, base, tch = [], [], []
    agree = []
    for i, (path, _) in enumerate(paths):
        gt = blob["gt%d" % i].astype(np.int8)
        runs = pm.valid_runs(gt)
        pred = drop_short_dead(decode(blob["p%d" % i], runs, params), runs, args.short_dead)
        df = read_features(path)[["since_touch_log", "last_touch_sign"]]
        at = np.flatnonzero(np.isclose(df["since_touch_log"].to_numpy(), 0.0))
        sign = df["last_touch_sign"].to_numpy()[at]
        live = (gt[at] == H) | (gt[at] == A)
        agree.append(np.mean((sign[live] > 0) == (gt[at][live] == H)))
        gts.append(gt)
        base.append((pred, runs))
        tch.append(df)
    home_sign = 1.0 if np.mean(agree) >= 0.5 else -1.0
    n_touch = sum(len(touches(df, home_sign)[0]) for df in tch)
    minutes = sum(len(g) for g in gts) / 25 / 60
    print("toques %.3f/s; acuerdo del signo con la GT (+1 = local): %.3f -> signo local %+d" % (
        n_touch / minutes / 60, float(np.mean(agree)), home_sign))
    ref = [p for p, _ in base]
    ref_pool = pooled(gts, ref)["spell_f1_mean"]
    ref_games = np.array([pm.evaluate(g, p)["spell_f1_mean"] for g, p in zip(gts, ref)])
    print("referencia (decodificacion %s + regla): %.4f" % (args.decode, ref_pool))
    for restarts in (False, True):
        for w in (12, 25, 50):
            preds = []
            for (pred, runs), df in zip(base, tch):
                tf, tt = touches(df, home_sign)
                preds.append(snap(pred, runs, tf, tt, w, restarts))
            score = pooled(gts, preds)["spell_f1_mean"]
            games = np.array([pm.evaluate(g, p)["spell_f1_mean"] for g, p in zip(gts, preds)])
            d = games - ref_games
            print("%-24s w=%2d: %.4f (%+.4f), mejor en %d, peor en %d, peor partido %+.4f" % (
                "directos + reanudaciones" if restarts else "solo cambios directos", w, score, score - ref_pool,
                int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9)), float(d.min())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
