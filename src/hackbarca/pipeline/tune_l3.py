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


def load_bench(use_tags: bool = True, version: int = 1, params=None):
    games = []
    for i, game in enumerate(BENCH_GAMES):
        match, gt = load_metrica_bench(game)
        degraded = degrade(match, params or BENCH_DEGRADE, seed=i)
        windows = simulate_set_piece_windows(metrica_events(game), seed=i) if use_tags else None
        games.append((features_for(degraded, windows, version=version), gt))
    return games


def random_degrade(rng):
    """Variante del simulador alrededor de la calibrada (aleatorizacion del dominio, E015)."""
    from dataclasses import replace
    return replace(
        BENCH_DEGRADE,
        ball_smooth_sigma=float(rng.uniform(4, 14)),
        ball_corr_noise=float(rng.uniform(0.2, 1.2)),
        ball_corr_tau=float(rng.uniform(25, 100)),
        gap_median=float(BENCH_DEGRADE.gap_median * rng.uniform(0.7, 1.4)),
        ball_drop_rate=float(rng.uniform(0.35, 0.5)),
        dead_ball_visible=float(rng.uniform(0.4, 0.7)),
    )


def augmented_games(n_extra: int, version: int, seed: int = 100):
    """Por partido, `n_extra` degradaciones adicionales con parametros sorteados (solo entrenar)."""
    rng = np.random.default_rng(seed)
    extra = {}
    for i, game in enumerate(BENCH_GAMES):
        match, gt = load_metrica_bench(game)
        extra[i] = [(features_for(degrade(match, random_degrade(rng), seed=seed + 10 * i + k),
                                  None, version=version), gt) for k in range(n_extra)]
    return extra


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tags", action="store_true", help="ablation: sin Smart Tagging")
    parser.add_argument("--features", type=int, default=1, help="version de las features")
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-leaf", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--augment", type=int, default=0,
                        help="degradaciones extra por partido de entrenamiento (aleatorizacion del dominio)")
    parser.add_argument("--stack", action="store_true", help="segunda capa con contexto temporal (E018)")
    parser.add_argument("--eval-shift", action="store_true",
                        help="robustez: el partido retenido se degrada con la calibracion alternativa del balon")
    args = parser.parse_args(argv)

    from hackbarca.models import l3
    if args.num_leaves:
        l3.LGB_PARAMS["num_leaves"] = args.num_leaves
    if args.min_leaf:
        l3.LGB_PARAMS["min_data_in_leaf"] = args.min_leaf
    if args.rounds:
        l3.N_ROUNDS = args.rounds

    games = load_bench(use_tags=not args.no_tags, version=args.features)
    if args.eval_shift:
        from dataclasses import replace
        shifted = replace(BENCH_DEGRADE, ball_smooth_sigma=6.0, ball_corr_noise=0.3, ball_corr_tau=25.0)
        eval_games = load_bench(use_tags=not args.no_tags, version=args.features, params=shifted)
    else:
        eval_games = games
    gts = [gt for _, gt in games]
    runs = [pm.valid_runs(gt) for gt in gts]
    extra = augmented_games(args.augment, args.features) if args.augment else {}
    oof = []
    for g in range(len(games)):
        others = [o for o in range(len(games)) if o != g]
        train_sets = [games[o] for o in others] + [e for o in others for e in extra.get(o, [])]
        if args.stack:
            from hackbarca.models.l3 import predict_stacked, train_stacked
            models = train_stacked([df for df, _ in train_sets], [gt for _, gt in train_sets])
            oof.append(predict_stacked(models, eval_games[g][0]))
            model = models[1]
        else:
            model = train([df for df, _ in train_sets], [gt for _, gt in train_sets])
            oof.append(predict_proba(model, eval_games[g][0]))
        if g == 0:
            imp = sorted(zip(model.feature_importance("gain"), model.feature_name()), reverse=True)
            print("importancia (gain) top 15: " + ", ".join(n for _, n in imp[:15]))
    name = "oof_l3_notags.npz" if args.no_tags else "oof_l3.npz"
    if args.num_leaves or args.min_leaf or args.rounds:
        name = name.replace(".npz", "_nl{}_ml{}_r{}.npz".format(args.num_leaves, args.min_leaf, args.rounds))
    if args.features != 1:
        name = name.replace(".npz", "_f{}.npz".format(args.features))
    if args.augment:
        name = name.replace(".npz", "_aug{}.npz".format(args.augment))
    if args.eval_shift:
        name = name.replace(".npz", "_shift.npz")
    if args.stack:
        name = name.replace(".npz", "_stack.npz")
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
