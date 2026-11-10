"""E111: variantes de la decodificacion sobre OOF ya guardados (sin entrenar).

Idea de otro equipo (FCB-Hacks, docs/investigacion/07_competidores.md, I1): un traspaso H<->A debe costar
menos que una parada + reanudacion (H->D->A). Con la decodificacion de la entrega (cambio 20, a D 10, desde
D 10) los dos caminos cuestan lo mismo, y el Viterbi no penaliza "traspasar por un D corto", que es el
defecto del reto (E041-E043, ADR-0021).

Se mide, con la seleccion dejando un partido fuera de tune_l3 (ADR-0008):
1. "libre": los 41 candidatos de tune_l3 (misma semilla). Reproduce su agregado (control del script).
2. "libre+": los 41 + POOL candidatos sorteados igual (para separar el efecto de una rejilla mayor).
3. "m=k": los de libre+ con to_dead + from_dead >= switch + k.
4. Decodificaciones fijas cercanas a la de la entrega.
Cada una sin regla y con la regla de D cortos de la entrega (ADR-0021, 75 frames), en el banco tal cual y
con el exceso de D cortos del reto inyectado (`inject_short_dead`, tasas de short_dead_rule.py).

    uv run python -m hackbarca.pipeline.decode_variants --oof <fichero de external_data/>

El Viterbi va por lotes de candidatos (mismas operaciones y mismo orden que decode/viterbi.py, asi que
el camino es identico; se comprueba con --check).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hackbarca.eval.bench import pm, pooled
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.failure_analysis import drop_short_dead, inject_short_dead
from hackbarca.pipeline.short_dead_rule import EXCESS_OTHER, EXCESS_SAME
from hackbarca.pipeline.tune_l3 import sample_decode

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, A, D = 0, 1, 2
SHORT_DEAD = 75
MARGINS = (0, 5, 10, 20)
FIXED = [DecodeParams(),
         DecodeParams(switch=20.0, to_dead=15.0, from_dead=15.0),
         DecodeParams(switch=20.0, to_dead=20.0, from_dead=20.0),
         DecodeParams(switch=15.0, to_dead=20.0, from_dead=20.0),
         DecodeParams(switch=20.0, to_dead=25.0, from_dead=25.0),
         DecodeParams(switch=30.0, to_dead=20.0, from_dead=20.0)]


def _trans(params: DecodeParams) -> np.ndarray:
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = params.switch
    c[H, D] = c[A, D] = params.to_dead
    c[D, H] = c[D, A] = params.from_dead
    return c


def viterbi_batch(logp: np.ndarray, temps: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(C, n) caminos. logp = log(p + 1e-6) (n, 3); temps (C,); trans (C, 3, 3).

    Igual que decode/viterbi.viterbi con emission = -temp * logp, candidato a candidato."""
    n, c = logp.shape[0], len(temps)
    em = -temps[:, None, None] * logp[None]                      # (C, n, 3)
    back = np.empty((n, c, 3), dtype=np.int8)
    acc = em[:, 0].copy()
    rows = np.arange(c)[:, None]
    cols = np.arange(3)[None, :]
    for t in range(1, n):
        total = acc[:, :, None] + trans                          # (C, prev, cur)
        b = np.argmin(total, axis=1)                             # (C, cur)
        back[t] = b
        acc = total[rows, b, cols] + em[:, t]
    path = np.empty((c, n), dtype=np.int8)
    path[:, -1] = np.argmin(acc, axis=1)
    for t in range(n - 1, 0, -1):
        path[:, t - 1] = back[t, np.arange(c), path[:, t]]
    return path


def decode_batch(proba: np.ndarray, runs, cands, batch: int = 40) -> np.ndarray:
    """(C, n) decodificaciones de `proba` para cada DecodeParams de `cands` (fijas, sin alphas)."""
    logp = np.log(proba + 1e-6)
    out = np.full((len(cands), len(proba)), D, dtype=np.int8)
    for lo in range(0, len(cands), batch):
        chunk = cands[lo:lo + batch]
        temps = np.array([c.temperature for c in chunk])
        trans = np.stack([_trans(c) for c in chunk])
        for s, e in runs:
            out[lo:lo + len(chunk), s:e] = viterbi_batch(logp[s:e], temps, trans)
    return out


def candidates(seed: int, pool: int):
    rng = np.random.default_rng(seed)
    base = [DecodeParams()] + [sample_decode(rng) for _ in range(40)]   # == tune_l3 --trials 40
    rng2 = np.random.default_rng(seed + 1000)
    extra = [sample_decode(rng2) for _ in range(pool)]
    return base, extra


def logo(scores: np.ndarray, subset) -> list:
    """Para cada partido retenido, el candidato del subconjunto con mejor media en los otros 9."""
    subset = list(subset)
    chosen = []
    for g in range(scores.shape[1]):
        others = [o for o in range(scores.shape[1]) if o != g]
        chosen.append(subset[int(np.argmax(scores[subset][:, others].mean(1)))])
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="fichero en external_data/")
    parser.add_argument("--pool", type=int, default=160, help="candidatos extra sorteados (libre+ y m=k)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5, help="semillas de la inyeccion del defecto del reto")
    parser.add_argument("--check", action="store_true", help="compara el Viterbi por lotes con decode()")
    args = parser.parse_args(argv)

    blob = np.load(os.path.join(_REPO_ROOT, "external_data", args.oof))
    n_games = len([k for k in blob.files if k.startswith("p")])
    probas = [blob["p%d" % i] for i in range(n_games)]
    gts = [blob["gt%d" % i].astype(np.int8) for i in range(n_games)]
    runs = [pm.valid_runs(g) for g in gts]

    base, extra = candidates(args.seed, args.pool)
    cands = base + extra + FIXED
    i_base = list(range(len(base)))
    i_plus = list(range(len(base) + len(extra)))
    i_fixed = list(range(len(base) + len(extra), len(cands)))

    if args.check:
        for c in (0, 7, len(cands) - 2):
            ref = decode(probas[0], runs[0], cands[c])
            got = decode_batch(probas[0], runs[0], [cands[c]])[0]
            print("check candidato %d: %s" % (c, "identico" if np.array_equal(ref, got) else "DISTINTO"))

    # decodificaciones y notas de cada candidato, sin regla y con la regla de D cortos
    preds = {}
    clean = np.zeros((2, len(cands), n_games))
    for g in range(n_games):
        paths = decode_batch(probas[g], runs[g], cands)
        for c in range(len(cands)):
            raw = paths[c]
            rule = drop_short_dead(raw, runs[g], SHORT_DEAD)
            preds[0, c, g], preds[1, c, g] = raw, rule
            clean[0, c, g] = pm.evaluate(gts[g], raw)["spell_f1_mean"]
            clean[1, c, g] = pm.evaluate(gts[g], rule)["spell_f1_mean"]
        print("partido %d de %d decodificado" % (g + 1, n_games), flush=True)

    def injected(chosen, r):
        """spellF1 agregada con el exceso de D cortos del reto inyectado (media de semillas)."""
        vals = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            out = []
            for g in range(n_games):
                inj = inject_short_dead(preds[0, chosen[g], g], runs[g], EXCESS_SAME, EXCESS_OTHER, rng)
                out.append(drop_short_dead(inj, runs[g], SHORT_DEAD) if r else inj)
            vals.append(pooled(gts, out)["spell_f1_mean"])
        return float(np.mean(vals))

    def report(label, chosen, ref_per=None):
        rows = []
        for r, rname in ((0, "sin regla"), (1, "regla 3 s")):
            held = [preds[r, chosen[g], g] for g in range(n_games)]
            res = pooled(gts, held)
            per = np.array([clean[r, chosen[g], g] for g in range(n_games)])
            line = "%-22s %-9s banco %.4f (poss %d) | inyectado %.4f" % (
                label, rname, res["spell_f1_mean"], res["n_pred_spells"], injected(chosen, r))
            if ref_per is not None:
                d = per - ref_per[r]
                line += " | frente a libre: %+.4f, mejor %d, peor %d, peor partido %+.4f" % (
                    res["spell_f1_mean"] - ref_pooled[r], int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9)), d.min())
            rows.append((line, per))
            print(line)
        return [p for _, p in rows]

    print("GT: %d posesiones; candidatos: libre %d, libre+ %d, fijos %d" % (
        pooled(gts, gts)["n_gt_spells"], len(i_base), len(i_plus), len(i_fixed)))
    chosen_free = [logo(clean[r], i_base) for r in (0, 1)]
    ref_pooled = [pooled(gts, [preds[r, chosen_free[r][g], g] for g in range(n_games)])["spell_f1_mean"]
                  for r in (0, 1)]
    ref_per = [np.array([clean[r, chosen_free[r][g], g] for g in range(n_games)]) for r in (0, 1)]
    print("== seleccion dejando un partido fuera (la regla, si va, entra antes de elegir)")
    for r in (0, 1):
        held = [preds[r, chosen_free[r][g], g] for g in range(n_games)]
        print("%-22s %-9s banco %.4f | inyectado %.4f | por partido %s" % (
            "libre (tune_l3)", ("sin regla", "regla 3 s")[r], pooled(gts, held)["spell_f1_mean"],
            injected(chosen_free[r], r), " ".join("%.3f" % x for x in ref_per[r])))
    variants = [("libre+", i_plus)] + [
        ("m=%d" % m, [c for c in i_plus if cands[c].to_dead + cands[c].from_dead >= cands[c].switch + m])
        for m in MARGINS]
    for label, subset in variants:
        if not subset:
            print("%s: sin candidatos" % label)
            continue
        for r in (0, 1):
            chosen = logo(clean[r], subset)
            held = [preds[r, chosen[g], g] for g in range(n_games)]
            per = np.array([clean[r, chosen[g], g] for g in range(n_games)])
            d = per - ref_per[r]
            picks = sorted({(round(cands[c].temperature, 2), round(cands[c].switch, 1), round(cands[c].to_dead, 1),
                             round(cands[c].from_dead, 1)) for c in chosen})
            print("%-22s %-9s banco %.4f | inyectado %.4f | frente a libre %+.4f, mejor %d, peor %d, peor partido "
                  "%+.4f | n=%d | elegidos (T, cambio, a D, desde D): %s" % (
                      label, ("sin regla", "regla 3 s")[r], pooled(gts, held)["spell_f1_mean"], injected(chosen, r),
                      pooled(gts, held)["spell_f1_mean"] - ref_pooled[r], int(np.sum(d > 1e-9)),
                      int(np.sum(d < -1e-9)), d.min(), len(subset), picks[:4]))
    print("== decodificaciones fijas (frente a la de la entrega 1/20/10/10)")
    entrega = i_fixed[0]
    for c in i_fixed:
        p = cands[c]
        for r in (0, 1):
            held = [preds[r, c, g] for g in range(n_games)]
            d = clean[r, c] - clean[r, entrega]
            print("T %.2f cambio %5.1f a D %5.1f desde D %5.1f %-9s banco %.4f (poss %d) | inyectado %.4f | "
                  "frente a la entrega %+.4f, mejor %d, peor %d, peor partido %+.4f" % (
                      p.temperature, p.switch, p.to_dead, p.from_dead, ("sin regla", "regla 3 s")[r],
                      pooled(gts, held)["spell_f1_mean"], pooled(gts, held)["n_pred_spells"],
                      injected([c] * n_games, r), pooled(gts, held)["spell_f1_mean"]
                      - pooled(gts, [preds[r, entrega, g] for g in range(n_games)])["spell_f1_mean"],
                      int(np.sum(d > 1e-9)), int(np.sum(d < -1e-9)), d.min()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
