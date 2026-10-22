"""L3 en el banco externo: LightGBM dejando un partido fuera + ajuste de la decodificacion.

    uv run python -m hackbarca.pipeline.tune_l3 --trials 40

1. Probabilidades fuera de muestra (OOF) de cada partido con un modelo entrenado en los otros.
2. Parametros de Viterbi elegidos en los OOF de dos partidos y evaluados en el tercero.
Guarda los OOF en external_data/oof_l3.npz para el analisis de errores (T8).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict

import numpy as np

from hackbarca.eval.bench import pm, pooled
from hackbarca.io.degrade import degrade
from hackbarca.io.external import (controlled_possession, is_dfl, load_bench_game,
                                   load_metrica_bench, metrica_events)
from hackbarca.io.smart_tagging import simulate_set_piece_windows
from hackbarca.models.l3 import DecodeParams, decode, features_for
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES, BENCH_GAMES_V3

BENCHES = {"v2": BENCH_GAMES, "v3": BENCH_GAMES_V3}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def feature_spec(text: str):
    """--features: un entero (versiones antiguas) o familias unidas por '+'."""
    return int(text) if str(text).isdigit() else str(text)


def sample_decode(rng) -> DecodeParams:
    return DecodeParams(
        temperature=float(np.exp(rng.uniform(np.log(0.05), np.log(2.0)))),
        switch=float(np.exp(rng.uniform(np.log(2), np.log(200)))),
        to_dead=float(np.exp(rng.uniform(np.log(2), np.log(200)))),
        from_dead=float(np.exp(rng.uniform(np.log(2), np.log(200)))),
    )


FEATURE_CACHE = os.path.join(_REPO_ROOT, "external_data", "features")
# el codigo del que dependen las features entra en la clave: si cambia, la cache se invalida sola
_FEATURE_CODE = ("features/frame_features.py", "features/tactical.py", "io/degrade.py", "models/l1.py")


def _code_hash() -> str:
    src = os.path.join(_REPO_ROOT, "src", "hackbarca")
    digest = hashlib.sha1()
    for rel in _FEATURE_CODE:
        with open(os.path.join(src, rel), "rb") as handle:
            digest.update(handle.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def feature_path(game: str, index: int, version: int, use_tags: bool, params) -> str:
    """Fichero de cache de las features de un partido degradado (semilla = posicion en el banco)."""
    key = hashlib.sha1(repr((game, index, version, use_tags, asdict(params), _code_hash())).encode())
    return os.path.join(FEATURE_CACHE, "{}_v{}_{}.npz".format(game, version, key.hexdigest()[:12]))


def read_features(path: str):
    import pandas as pd
    blob = np.load(path)
    return pd.DataFrame(blob["values"], columns=[str(c) for c in blob["columns"]])


def bench_feature_paths(use_tags: bool = True, version: int = 1, params=None, names=BENCH_GAMES):
    """[(ruta de features en cache, gt)] por partido; calcula y guarda solo lo que falte."""
    params = params or BENCH_DEGRADE
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    out = []
    for i, game in enumerate(names):
        match, gt = load_bench_game(game)
        path = feature_path(game, i, version, use_tags, params)
        if not os.path.exists(path):
            degraded = degrade(match, params, seed=i)
            # el Smart Tagging simulado sale de los eventos de Metrica; DFL va siempre sin etiquetas
            windows = (simulate_set_piece_windows(metrica_events(game), seed=i)
                       if use_tags and not is_dfl(game) else None)
            df = features_for(degraded, windows, version=version)
            tmp = path[:-4] + ".tmp.npz"  # escritura atomica: un proceso cortado no deja cache rota
            np.savez(tmp, columns=np.array(df.columns, dtype=str), values=df.to_numpy(np.float32))
            os.replace(tmp, path)
        out.append((path, gt))
    return out


def load_bench(use_tags: bool = True, version: int = 1, params=None, names=BENCH_GAMES):
    return [(read_features(path), gt)
            for path, gt in bench_feature_paths(use_tags, version, params, names)]


def _materialize(item):
    return read_features(item) if isinstance(item, str) else item


def _fit_predict(job):
    """Un pliegue: entrena con train_sets y da las probabilidades del partido retenido.

    Los partidos llegan como rutas de la cache (o DataFrames si son degradaciones extra): asi un
    worker no recibe copias de todo el banco y solo carga lo que usa.
    """
    items, eval_item, stack, lgb_params, n_rounds, dynamic = job
    train_sets = [(_materialize(x), gt) for x, gt in items]
    eval_df = _materialize(eval_item)
    from hackbarca.models import l3
    l3.LGB_PARAMS.update(lgb_params)
    l3.N_ROUNDS = n_rounds
    features, labels = [df for df, _ in train_sets], [gt for _, gt in train_sets]
    p_change = None
    if dynamic:  # E032: detector de cambio entrenado con los mismos partidos del pliegue
        from hackbarca.models.change import predict_change, train_change
        p_change = predict_change(train_change(features, labels), eval_df)
    if stack:
        models = l3.train_stacked(features, labels)
        return l3.predict_stacked(models, eval_df), models[1], p_change
    model = l3.train(features, labels)
    return l3.predict_proba(model, eval_df), model, p_change


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
    parser.add_argument("--features", type=feature_spec, default=1,
                        help="version antigua (3 = vigente) o familias: touch+ball+players+blocks+structure+restart")
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-leaf", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--augment", type=int, default=0,
                        help="degradaciones extra por partido de entrenamiento (aleatorizacion del dominio)")
    parser.add_argument("--stack", action="store_true", help="segunda capa con contexto temporal (E018)")
    parser.add_argument("--eval-shift", action="store_true",
                        help="robustez: el partido retenido se degrada con la calibracion alternativa del balon")
    parser.add_argument("--controlled", type=float, default=0.0,
                        help="GT con posesion controlada: segundos minimos para que un robo cuente (E020)")
    parser.add_argument("--controlled-dfl", type=float, default=None,
                        help="T para los partidos DFL (por defecto el de --controlled); 2 s cuadra con la GT publica")
    parser.add_argument("--train-gt", choices=("controlled", "raw"), default="controlled",
                        help="con --controlled, entrenar tambien con la GT controlada (E021b) o con la original")
    parser.add_argument("--from-oof", default=None,
                        help="no entrena: re-decodifica los OOF guardados (external_data/<fichero>)")
    parser.add_argument("--bench", choices=sorted(BENCHES), default="v2",
                        help="v2: 3 Metrica; v3: 3 Metrica + 7 DFL")
    parser.add_argument("--jobs", type=int, default=1, help="pliegues en paralelo (4 hilos cada uno)")
    parser.add_argument("--dynamic", action="store_true",
                        help="detector de cambio + Viterbi con coste por frame (E032)")
    args = parser.parse_args(argv)
    names = BENCHES[args.bench]

    from hackbarca.models import l3
    if args.num_leaves:
        l3.LGB_PARAMS["num_leaves"] = args.num_leaves
    if args.min_leaf:
        l3.LGB_PARAMS["min_data_in_leaf"] = args.min_leaf
    if args.rounds:
        l3.N_ROUNDS = args.rounds

    if args.from_oof:
        games = [(None, load_bench_game(game)[1]) for game in names]
    else:
        # rutas de la cache, no DataFrames: el proceso principal no guarda las features en memoria
        games = bench_feature_paths(use_tags=not args.no_tags, version=args.features, names=names)
    train_games = games
    if args.controlled:
        def t_of(game):
            t = args.controlled_dfl if (is_dfl(game) and args.controlled_dfl is not None) else args.controlled
            return int(round(t * 25))
        games = [(df, controlled_possession(gt, t_of(n))) for (df, gt), n in zip(games, names)]
        if args.train_gt == "controlled":
            train_games = games
    if args.eval_shift:
        from dataclasses import replace
        shifted = replace(BENCH_DEGRADE, ball_smooth_sigma=6.0, ball_corr_noise=0.3, ball_corr_tau=25.0)
        eval_games = bench_feature_paths(use_tags=not args.no_tags, version=args.features, params=shifted,
                                         names=names)
    else:
        eval_games = games
    gts = [gt for _, gt in games]
    runs = [pm.valid_runs(gt) for gt in gts]
    extra = augmented_games(args.augment, args.features) if args.augment else {}
    oof = []
    if args.from_oof:
        blob = np.load(os.path.join(_REPO_ROOT, "external_data", args.from_oof))
        oof = [blob["p%d" % i] for i in range(len(games))]
        changes = [blob["c%d" % i] for i in range(len(games))] if "c0" in blob.files else None
    else:
        changes = None
    jobs = []
    for g in range(len(games) if not args.from_oof else 0):
        others = [o for o in range(len(games)) if o != g]
        train_sets = [train_games[o] for o in others] + [e for o in others for e in extra.get(o, [])]
        jobs.append((train_sets, eval_games[g][0], args.stack, dict(l3.LGB_PARAMS), l3.N_ROUNDS, args.dynamic))
    if jobs:
        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                results = list(pool.map(_fit_predict, jobs))
        else:
            results = [_fit_predict(job) for job in jobs]
        oof = [p for p, _, _ in results]
        changes = [c for _, _, c in results] if args.dynamic else None
        model = results[0][1]
        imp = sorted(zip(model.feature_importance("gain"), model.feature_name()), reverse=True)
        print("importancia (gain) top 15: " + ", ".join(n for _, n in imp[:15]))
    name = "oof_l3_notags.npz" if args.no_tags else "oof_l3.npz"
    if args.bench != "v2":
        name = name.replace(".npz", "_{}.npz".format(args.bench))
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
    if args.dynamic:
        name = name.replace(".npz", "_dyn.npz")
    if args.controlled:
        name = name.replace(".npz", "_ctl{:g}.npz".format(args.controlled))
        if args.controlled_dfl is not None:
            name = name.replace(".npz", "_dfl{:g}.npz".format(args.controlled_dfl))
        if args.train_gt == "raw":
            name = name.replace(".npz", "_trainraw.npz")
    if not args.from_oof:
        np.savez_compressed(os.path.join(_REPO_ROOT, "external_data", name),
                            **{"p%d" % i: p for i, p in enumerate(oof)}, **{"gt%d" % i: g for i, g in enumerate(gts)},
                            **({"c%d" % i: c for i, c in enumerate(changes)} if changes else {}))

    argmax = [np.argmax(p, 1).astype(np.int8) for p in oof]
    r = pooled(gts, argmax)
    print("L3 argmax (sin Viterbi): spellF1={:.4f} macroF1={:.4f} poss={}".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))

    rng = np.random.default_rng(args.seed)
    trials = [DecodeParams()] + [sample_decode(rng) for _ in range(args.trials)]
    if changes is not None:
        # mismos trials fijos (comparables con el Viterbi de siempre) + una copia con alphas sorteados
        from dataclasses import replace
        rng_dyn = np.random.default_rng(args.seed + 1)
        trials += [replace(t, alpha_switch=float(rng_dyn.uniform(0, 1.5)),
                           alpha_dead=float(rng_dyn.uniform(0, 1.5))) for t in list(trials)]
    scores = np.zeros((len(trials), len(games)))
    preds = {}
    for t, params in enumerate(trials):
        for g in range(len(games)):
            preds[t, g] = decode(oof[g], runs[g], params, changes[g] if changes is not None else None)
            scores[t, g] = pm.evaluate(gts[g], preds[t, g])["spell_f1_mean"]
    held = []
    for g in range(len(games)):
        others = [o for o in range(len(games)) if o != g]
        best = int(np.argmax(scores[:, others].mean(1)))
        held.append(preds[best, g])
        print("retenido game {}: spellF1={:.4f} {}".format(names[g], scores[best, g], asdict(trials[best])))
    r = pooled(gts, held)
    print("L3 LOGO agregado: spellF1={:.4f} macroF1={:.4f} poss={} (gt {})".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    best_all = int(np.argmax(scores.mean(1)))
    print("mejor decodificacion en todos: {} media={:.4f}".format(asdict(trials[best_all]), scores[best_all].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
