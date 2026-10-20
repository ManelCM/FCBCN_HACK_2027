"""L3 en el banco externo: LightGBM dejando un partido fuera + ajuste de la decodificacion.

    uv run python -m hackbarca.pipeline.tune_l3 --trials 40

1. Probabilidades fuera de muestra (OOF) de cada partido con un modelo entrenado en los otros.
2. Parametros de Viterbi elegidos en los OOF de dos partidos y evaluados en el tercero.
Guarda los OOF en external_data/oof_l3.npz para el analisis de errores (T8).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np

from hackbarca.eval.bench import pm, pooled
from hackbarca.io.degrade import degrade
from hackbarca.io.external import load_metrica_bench, metrica_events
from hackbarca.io.smart_tagging import simulate_set_piece_windows
from hackbarca.models.l3 import DecodeParams, decode, features_for, predict_proba, train
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def sample_decode(rng) -> DecodeParams:
    return DecodeParams(
        temperature=float(np.exp(rng.uniform(np.log(0.05), np.log(2.0)))),
        switch=float(np.exp(rng.uniform(np.log(2), np.log(200)))),
        to_dead=float(np.exp(rng.uniform(np.log(2), np.log(200)))),
        from_dead=float(np.exp(rng.uniform(np.log(2), np.log(200)))),
    )


def load_bench(use_tags: bool = True):
    games = []
    for i, game in enumerate(BENCH_GAMES):
        match, gt = load_metrica_bench(game)
        degraded = degrade(match, BENCH_DEGRADE, seed=i)
        windows = simulate_set_piece_windows(metrica_events(game), seed=i) if use_tags else None
        games.append((features_for(degraded, windows), gt))
    return games


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tags", action="store_true", help="ablation: sin Smart Tagging")
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-leaf", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args(argv)

    from hackbarca.models import l3
    if args.num_leaves:
        l3.LGB_PARAMS["num_leaves"] = args.num_leaves
    if args.min_leaf:
        l3.LGB_PARAMS["min_data_in_leaf"] = args.min_leaf
    if args.rounds:
        l3.N_ROUNDS = args.rounds

    games = load_bench(use_tags=not args.no_tags)
    gts = [gt for _, gt in games]
    runs = [pm.valid_runs(gt) for gt in gts]
    oof = []
    for g in range(len(games)):
        others = [o for o in range(len(games)) if o != g]
        model = train([games[o][0] for o in others], [gts[o] for o in others])
        oof.append(predict_proba(model, games[g][0]))
        if g == 0:
            imp = sorted(zip(model.feature_importance("gain"), model.feature_name()), reverse=True)
            print("importancia (gain) top 15: " + ", ".join(n for _, n in imp[:15]))
    name = "oof_l3_notags.npz" if args.no_tags else "oof_l3.npz"
    if args.num_leaves or args.min_leaf or args.rounds:
        name = name.replace(".npz", "_nl{}_ml{}_r{}.npz".format(args.num_leaves, args.min_leaf, args.rounds))
    np.savez_compressed(os.path.join(_REPO_ROOT, "external_data", name),
                        **{"p%d" % i: p for i, p in enumerate(oof)}, **{"gt%d" % i: g for i, g in enumerate(gts)})

    argmax = [np.argmax(p, 1).astype(np.int8) for p in oof]
    r = pooled(gts, argmax)
    print("L3 argmax (sin Viterbi): spellF1={:.4f} macroF1={:.4f} poss={}".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))

    rng = np.random.default_rng(args.seed)
    trials = [DecodeParams()] + [sample_decode(rng) for _ in range(args.trials)]
    scores = np.zeros((len(trials), len(games)))
    preds = {}
    for t, params in enumerate(trials):
        for g in range(len(games)):
            preds[t, g] = decode(oof[g], runs[g], params)
            scores[t, g] = pm.evaluate(gts[g], preds[t, g])["spell_f1_mean"]
    held = []
    for g in range(len(games)):
        others = [o for o in range(len(games)) if o != g]
        best = int(np.argmax(scores[:, others].mean(1)))
        held.append(preds[best, g])
        print("retenido game {}: spellF1={:.4f} {}".format(BENCH_GAMES[g], scores[best, g], asdict(trials[best])))
    r = pooled(gts, held)
    print("L3 LOGO agregado: spellF1={:.4f} macroF1={:.4f} poss={} (gt {})".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    best_all = int(np.argmax(scores.mean(1)))
    print("mejor decodificacion en los 3: {} media={:.4f}".format(asdict(trials[best_all]), scores[best_all].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
