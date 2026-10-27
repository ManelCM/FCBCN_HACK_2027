"""E090: regla de D cortos sobre la decodificacion de la entrega, medida en el banco v3.1.

En el reto el modelo predice 0,53 D cortos (< 5 s) por minuto y la GT del banco tiene 0,09 (E075c): si
el reto se parece al banco, casi todos son errores. La regla `failure_analysis.drop_short_dead` (cada D
de menos de k s entre dos tramos vivos toma la etiqueta del tramo anterior) no se puede aceptar con el
banco limpio, porque alli casi no hay D cortos que quitar (E040: 0 a -0,008). Aqui se mide:

1. el coste en el banco tal cual, por partido (lo que la regla rompe donde el modelo acierta);
2. lo que recupera con el exceso de D cortos del reto inyectado (`inject_short_dead`, tasas del reto
   menos las del banco). Es circular a proposito: dice cuanto vale la regla SI los D cortos de mas del
   reto son errores como los inyectados, no si lo son.

    uv run python -m hackbarca.pipeline.short_dead_rule --oof <oof del banco v3.1>
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hackbarca.eval.bench import pm
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
from hackbarca.pipeline.failure_analysis import drop_short_dead, inject_short_dead, spell

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KS = (0, 25, 50, 75, 100, 125)   # frames; 0 = sin regla
# exceso del reto sobre nuestra prediccion en el banco, por minuto (E075c: 0,246/0,283 frente a 0,056/0,069)
EXCESS_SAME, EXCESS_OTHER = 0.19, 0.21


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="fichero en external_data/")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)
    blob = np.load(os.path.join(_REPO_ROOT, "external_data", args.oof))
    gts, preds = [], []
    for i in range(len(BENCH_GAMES_V3)):
        gt = blob["gt%d" % i].astype(np.int8)
        gts.append(gt)
        preds.append(decode(blob["p%d" % i], pm.valid_runs(gt), DecodeParams()))
    base = [spell([g], [p])["mean_f1"] for g, p in zip(gts, preds)]
    print("decodificacion de la entrega (%s): %.4f" % (DecodeParams(), spell(gts, preds)["mean_f1"]))
    for k in KS[1:]:
        fixed = [drop_short_dead(p, pm.valid_runs(g), k) for g, p in zip(gts, preds)]
        per = [spell([g], [f])["mean_f1"] for g, f in zip(gts, fixed)]
        diff = np.array(per) - np.array(base)
        print("regla D < %.0f s, banco tal cual: %.4f (%+.4f) | partidos mejor %d, peor %d | peor partido %+.4f" % (
            k / 25, spell(gts, fixed)["mean_f1"], spell(gts, fixed)["mean_f1"] - spell(gts, preds)["mean_f1"],
            np.sum(diff > 1e-9), np.sum(diff < -1e-9), diff.min()))
    for k in KS:
        vals = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            inj = [inject_short_dead(p, pm.valid_runs(g), EXCESS_SAME, EXCESS_OTHER, rng) for g, p in zip(gts, preds)]
            if k:
                inj = [drop_short_dead(p, pm.valid_runs(g), k) for g, p in zip(gts, inj)]
            vals.append(spell(gts, inj)["mean_f1"])
        print("con el exceso del reto inyectado, %s: %.4f" % (
            "sin regla" if not k else "regla D < %.0f s" % (k / 25), np.mean(vals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
