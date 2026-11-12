"""E121 (I8 de otros equipos, katenaball/AndaAnda): mezclar en logit el L3 con una red temporal.

E079b ya promedió las probabilidades del MS-TCN (E079) con E049 y no pasó. Aquí se mezcla en logit
(media geométrica normalizada, p ∝ p_l3^(1-w) · p_red^w) con el OOF de la base vigente y la regla de 3 s.
w se elige dejando un partido fuera, entre {0, 0,15, 0,3, 0,5}, con la decodificación fijada de la base.

    uv run python -m hackbarca.pipeline.logit_blend --base <OOF> --deep oof_mstcn_v3_1.npz --decode T,c,aD,dD
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hackbarca.eval.bench import pm, pooled
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.failure_analysis import drop_short_dead

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WEIGHTS = (0.0, 0.15, 0.3, 0.5)


def blend(p: np.ndarray, q: np.ndarray, w: float) -> np.ndarray:
    logp = (1 - w) * np.log(np.clip(p, 1e-6, 1)) + w * np.log(np.clip(q, 1e-6, 1))
    out = np.exp(logp - logp.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--deep", default="oof_mstcn_v3_1.npz")
    parser.add_argument("--decode", default="1,20,10,10")
    parser.add_argument("--short-dead", type=int, default=75)
    args = parser.parse_args(argv)
    params = DecodeParams(*(float(x) for x in args.decode.split(",")))
    a = np.load(os.path.join(_REPO_ROOT, "external_data", args.base))
    b = np.load(os.path.join(_REPO_ROOT, "external_data", args.deep))
    n = len([k for k in a.files if k.startswith("gt")])
    gts, preds = [], {w: [] for w in WEIGHTS}
    for i in range(n):
        gt = a["gt%d" % i].astype(np.int8)
        assert len(gt) == len(b["gt%d" % i]), "OOF de partidos distintos"   # la GT puntuada es la de la base
        runs = pm.valid_runs(gt)
        gts.append(gt)
        for w in WEIGHTS:
            p = blend(a["p%d" % i], b["p%d" % i], w)
            preds[w].append(drop_short_dead(decode(p, runs, params), runs, args.short_dead))
    score = {w: np.array([pm.evaluate(g, p)["spell_f1_mean"] for g, p in zip(gts, preds[w])]) for w in WEIGHTS}
    for w in WEIGHTS:
        d = score[w] - score[0.0]
        print("w=%.2f: %.4f (%+.4f), mejor en %d de %d | por partido %s" % (
            w, pooled(gts, preds[w])["spell_f1_mean"], pooled(gts, preds[w])["spell_f1_mean"]
            - pooled(gts, preds[0.0])["spell_f1_mean"], int(np.sum(d > 1e-9)), n,
            " ".join("%.3f" % x for x in score[w])))
    chosen = []   # dejando un partido fuera: w con la mejor media de spellF1 en los otros
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        chosen.append(max(WEIGHTS, key=lambda w: score[w][rest].mean()))
    logo = [preds[w][i] for i, w in enumerate(chosen)]
    s = pooled(gts, logo)["spell_f1_mean"]
    d = np.array([pm.evaluate(g, p)["spell_f1_mean"] for g, p in zip(gts, logo)]) - score[0.0]
    print("LOGO: %.4f (%+.4f frente a w=0), mejor en %d de %d; w elegidos %s" % (
        s, s - pooled(gts, preds[0.0])["spell_f1_mean"], int(np.sum(d > 1e-9)), n, chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
