"""Poseedor (M5a, E078) en el banco v3.1: probabilidades fuera de muestra y puerta.

    uv run --group vision python -m hackbarca.pipeline.tune_possessor [--folds 0,3,7]

Entradas por partido degradado con v3.1 (misma semilla que el banco) en external_data/possessor/.
Puerta (plan completo, en 3 pliegues): promediado con el LightGBM de la misma configuracion
(OOF de --base-oof), cambios H<->A con frontera a <= 2 s >= 70 % (hoy 63 %) y sobrantes por debajo
de los del LightGBM solo. Con --folds all hace los 10 pliegues y guarda external_data/oof_poss_v3_1.npz
para tune_l3 --deep-oof.
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.io.degrade import degrade
from hackbarca.io.external import controlled_possession, is_dfl, load_bench_game
from hackbarca.pipeline.bench_config import BENCH_DEGRADE_V3_1, BENCH_GAMES_V3
from hackbarca.pipeline.tune_l3 import _REPO_ROOT

CACHE = os.path.join(_REPO_ROOT, "external_data", "possessor")


def game_inputs(game: str, index: int):
    from hackbarca.models import possessor
    from hackbarca.models.seq import STEP
    src = [os.path.join(_REPO_ROOT, "src", "hackbarca", "models", f) for f in ("possessor.py", "seq.py")]
    h = hashlib.sha1(repr((game, index)).encode())
    for s in src:
        with open(s, "rb") as handle:
            h.update(handle.read().replace(b"\r\n", b"\n"))
    path = os.path.join(CACHE, "{}_{}.npz".format(game, h.hexdigest()[:10]))
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        match, gt = load_bench_game(game)
        deg = degrade(match, BENCH_DEGRADE_V3_1, seed=index)
        tokens, mask, glob, order = possessor.tokens_with_order(deg)
        poss = possessor.possessor_targets(match, deg, gt, order)
        sl = slice(0, None, STEP)
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, tokens=tokens[sl], mask=mask[sl], glob=glob[sl], gt=gt[sl], poss=poss[sl], n=match.n_frames)
        os.replace(tmp, path)
    b = np.load(path)
    return b["tokens"], b["mask"], b["glob"], b["gt"], b["poss"], int(b["n"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,3,7", help="indices de los partidos retenidos, o 'all'")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--base-oof", default=None, help="OOF del LightGBM (external_data/) para la puerta")
    parser.add_argument("--seed", type=int, default=None, help="semilla distinta (E125); el OOF lleva _s<semilla>")
    args = parser.parse_args(argv)
    from hackbarca.models.l3 import DecodeParams, decode
    from hackbarca.models.possessor import predict_possessor, train_possessor
    from hackbarca.pipeline.failure_analysis import change_masks, spell
    names = BENCH_GAMES_V3
    data = [game_inputs(g, i) for i, g in enumerate(names)]
    cov = np.mean(np.concatenate([d[4][d[4] >= 0] < 26 for d in data]))
    print("frames vivos con poseedor emparejado: %.3f" % cov, flush=True)
    folds = list(range(len(names))) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    base = np.load(os.path.join(_REPO_ROOT, "external_data", args.base_oof)) if args.base_oof else None
    oof = {}
    rows = []
    for g in folds:
        kw = {} if args.seed is None else {"seed": args.seed}
        model = train_possessor([d[:5] for o, d in enumerate(data) if o != g], epochs=args.epochs, **kw)
        tokens, mask, glob, _, _, n = data[g]
        p = predict_possessor(model, tokens, mask, glob, n)
        oof[g] = p
        match, raw = load_bench_game(names[g])
        gt = controlled_possession(raw, int(round((2.0 if is_dfl(names[g]) else 1.5) * 25)))
        runs = pm.valid_runs(gt)
        lab = gt != 3
        line = "retenido %s: acierto %.3f" % (names[g], np.mean(np.argmax(p, 1)[lab] == gt[lab]))
        if base is not None:
            pb = base["p%d" % g]
            for label, prob in (("lgbm", pb), ("poseedor", p), ("media", (pb + p) / 2)):
                pred = decode(prob, runs, DecodeParams())
                game = dict(gt=gt, pred=pred, known=~np.isnan(degrade(match, BENCH_DEGRADE_V3_1, seed=g).ball_x))
                _, _, _, c = change_masks(game)
                seen = c["vistos"] / max(c["vistos"] + c["no vistos, balon visible"] + c["no vistos, en hueco"], 1)
                rows.append((label, g, spell([gt], [pred])["mean_f1"], seen, c["sobrantes"]))
                line += " | %s spellF1 %.4f a<=2s %.3f sobrantes %d" % (label, rows[-1][2], seen, c["sobrantes"])
        print(line, flush=True)
    if rows:
        for label in ("lgbm", "poseedor", "media"):
            r = [x for x in rows if x[0] == label]
            print("%-9s spellF1 medio %.4f | cambios H<->A a <= 2 s %.3f | sobrantes %d" % (
                label, np.mean([x[2] for x in r]), np.mean([x[3] for x in r]), sum(x[4] for x in r)))
    if args.folds == "all":
        gts = [load_bench_game(nm)[1] for nm in names]
        out = os.path.join(_REPO_ROOT, "external_data", "oof_poss_v3_1%s.npz" % ("" if args.seed is None else "_s%d" % args.seed))
        np.savez_compressed(out, **{"p%d" % i: oof[i] for i in range(len(names))}, **{"gt%d" % i: g for i, g in enumerate(gts)})
        print("guardado", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
