"""E099b: la "retencion minima" del candidato de 40 features, fijada de antemano, sobre dos OOF del banco.

El informe externo eligio una retencion de 50 frames (2 s) con una busqueda LOGO entre 246
combinaciones de decodificacion y retencion. Aqui no se busca nada: se fija k = 50 y se aplica igual a
la referencia y al candidato, sobre la decodificacion de la entrega (`DecodeParams()`, la de
models/artifacts/l3_decode.json) y con la regla de D cortos de la entrega (ADR-0021) antes o sin ella.

Retencion: dentro de cada tramo puntuado, todo tramo predicho (H, A o D) de menos de k frames toma la
etiqueta del tramo anterior (el primero, la del siguiente). A diferencia de la regla de D cortos, tambien
quita posesiones cortas, asi que en el reto puede borrar robos reales.

    uv run python -m hackbarca.pipeline.retention_compare --base <oof E075> --cand <oof E099>
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hackbarca.eval.bench import pm, pooled
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
from hackbarca.pipeline.failure_analysis import drop_short_dead

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def drop_short_spells(pred: np.ndarray, runs, k: int) -> np.ndarray:
    """Todo tramo predicho de menos de k frames toma la etiqueta del anterior (o del siguiente)."""
    out = pred.copy()
    for s, e in runs:
        segs = pm.segments(out, [(s, e)])
        if len(segs) < 2:
            continue
        for n, (a, b, code) in enumerate(segs):
            if b - a < k:
                out[a:b] = out[a - 1] if n > 0 else segs[1][2]
    return out


def _variants(oof_name: str, keep: int, short_dead: int):
    blob = np.load(os.path.join(_REPO_ROOT, "external_data", oof_name))
    gts, out = [], {"entrega sin regla": [], "entrega": [], "entrega + retencion": [], "retencion sola": []}
    for i in range(len(BENCH_GAMES_V3)):
        gt = blob["gt%d" % i].astype(np.int8)
        runs = pm.valid_runs(gt)
        raw = decode(blob["p%d" % i], runs, DecodeParams())
        rule = drop_short_dead(raw, runs, short_dead)
        gts.append(gt)
        out["entrega sin regla"].append(raw)
        out["entrega"].append(rule)
        out["entrega + retencion"].append(drop_short_spells(rule, runs, keep))
        out["retencion sola"].append(drop_short_spells(raw, runs, keep))
    return gts, out


def _per_game(gts, preds):
    return np.array([pm.evaluate(g, p)["spell_f1_mean"] for g, p in zip(gts, preds)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="OOF de la referencia (external_data/)")
    parser.add_argument("--cand", required=True, help="OOF del candidato (external_data/)")
    parser.add_argument("--keep", type=int, default=50, help="retencion minima en frames (fijada: 50 = 2 s)")
    parser.add_argument("--short-dead", type=int, default=75, help="regla de D cortos de la entrega (ADR-0021)")
    args = parser.parse_args(argv)
    gts, base = _variants(args.base, args.keep, args.short_dead)
    gts_c, cand = _variants(args.cand, args.keep, args.short_dead)
    assert all(np.array_equal(a, b) for a, b in zip(gts, gts_c)), "los dos OOF no tienen la misma GT"
    names = list(BENCH_GAMES_V3)
    ref = _per_game(gts, base["entrega"])
    print("partidos: " + " ".join(names))
    print("%-22s | %-24s | %-24s | %s" % ("variante", "referencia", "candidato", "cand - ref (misma variante)"))
    for label in base:
        rb, rc = pooled(gts, base[label]), pooled(gts, cand[label])
        pb, pc = _per_game(gts, base[label]), _per_game(gts, cand[label])
        d = pc - pb
        print("%-22s | %.4f mF1 %.4f n=%d | %.4f mF1 %.4f n=%d | %+.4f, mejor en %d de 10" % (
            label, rb["spell_f1_mean"], rb["macro_f1"], rb["n_pred_spells"], rc["spell_f1_mean"],
            rc["macro_f1"], rc["n_pred_spells"], rc["spell_f1_mean"] - rb["spell_f1_mean"], int(np.sum(d > 0))))
        print("    por partido ref:  " + " ".join("%.3f" % x for x in pb))
        print("    por partido cand: " + " ".join("%.3f" % x for x in pc))
    for label in ("entrega + retencion", "retencion sola"):
        pc = _per_game(gts, cand[label])
        rc = pooled(gts, cand[label])["spell_f1_mean"]
        print("candidato %s frente a la entrega vigente (referencia, entrega): %+.4f, mejor en %d de 10" % (
            label, rc - pooled(gts, base["entrega"])["spell_f1_mean"], int(np.sum(pc - ref > 0))))
    print("GT: %d posesiones" % pooled(gts, base["entrega"])["n_gt_spells"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
