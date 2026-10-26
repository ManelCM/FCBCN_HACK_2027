"""MS-TCN (M5c, E079) en el banco v3.1: probabilidades fuera de muestra dejando un partido fuera.

    uv run --group vision python -m hackbarca.pipeline.tune_mstcn

Usa la cache de features de tune_l3 (banco v3.1, familias del modelo de la entrega). Guarda
external_data/oof_mstcn_v3_1.npz para combinarlo con tune_l3 --from-oof ... --deep-oof.
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.features.frame_features import swap_teams
from hackbarca.pipeline.tune_l3 import BENCH_PARAMS, BENCHES, _REPO_ROOT, bench_feature_paths, read_features

FEATURES = "touch+ball+players+blocks+structure+restart"


def main() -> int:
    from hackbarca.models.mstcn import predict_mstcn, prepare, train_mstcn
    names = BENCHES["v3_1"]
    paths = bench_feature_paths(use_tags=False, version=FEATURES, params=BENCH_PARAMS["v3_1"], names=names)
    dfs = [read_features(p) for p, _ in paths]
    gts = [gt for _, gt in paths]
    oof = []
    for g in range(len(names)):
        others = [o for o in range(len(names)) if o != g]
        stacked = np.concatenate([dfs[o].to_numpy(np.float32)[::5] for o in others])
        mu, sd = np.nanmean(stacked, 0), np.nanstd(stacked, 0) + 1e-6
        xs, ys = [], []
        for o in others:
            for df, y in ((dfs[o], gts[o]), (swap_teams(dfs[o]), np.where(gts[o] == 0, 1, np.where(gts[o] == 1, 0, gts[o])))):
                xs.append(prepare(df, mu, sd)[0])
                ys.append(y[::5])
        model = train_mstcn(xs, ys)
        x = prepare(dfs[g], mu, sd)[0]
        xw = prepare(swap_teams(dfs[g]), mu, sd)[0]
        p = predict_mstcn(model, x, xw, len(gts[g]))
        oof.append(p)
        lab = gts[g] != 3
        print("retenido %s: acierto %.3f" % (names[g], np.mean(np.argmax(p, 1)[lab] == gts[g][lab])), flush=True)
    out = os.path.join(_REPO_ROOT, "external_data", "oof_mstcn_v3_1.npz")
    np.savez_compressed(out, **{"p%d" % i: p for i, p in enumerate(oof)}, **{"gt%d" % i: g for i, g in enumerate(gts)})
    print("guardado", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
