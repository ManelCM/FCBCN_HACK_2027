"""Modelo profundo en el banco (E034): probabilidades fuera de muestra dejando un partido fuera.

    uv run --group vision python -m hackbarca.pipeline.tune_seq --bench v3

Guarda external_data/oof_seq_<bench>.npz (p0.., gt0..) para combinarlo con LightGBM:

    uv run python -m hackbarca.pipeline.tune_l3 --bench v3 --from-oof <oof_l3...> --deep-oof oof_seq_v3.npz ...

Las entradas por partido (degradado como en tune_l3, semilla = posicion en el banco) se guardan en
external_data/seq/ para no recalcularlas en cada pliegue. Solo datos externos.
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

from hackbarca.io.degrade import degrade
from hackbarca.io.external import load_bench_game
from hackbarca.pipeline.bench_config import BENCH_DEGRADE
from hackbarca.pipeline.tune_l3 import BENCHES, _REPO_ROOT

CACHE = os.path.join(_REPO_ROOT, "external_data", "seq")


def game_inputs(game: str, index: int):
    """(tokens, mask, glob, gt, ball) a 5 fps del partido degradado, con cache en disco."""
    from hackbarca.models import seq
    src = os.path.join(_REPO_ROOT, "src", "hackbarca", "models", "seq.py")
    with open(src, "rb") as handle:
        key = hashlib.sha1(handle.read().replace(b"\r\n", b"\n") + repr((game, index)).encode()).hexdigest()[:10]
    path = os.path.join(CACHE, "{}_{}.npz".format(game, key))
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        match, gt = load_bench_game(game)
        parts = seq.seq_inputs(degrade(match, BENCH_DEGRADE, seed=index), gt, match)
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, tokens=parts[0], mask=parts[1], glob=parts[2], gt=parts[3], ball=parts[4], n=match.n_frames)
        os.replace(tmp, path)
    blob = np.load(path)
    return blob["tokens"], blob["mask"], blob["glob"], blob["gt"], blob["ball"], int(blob["n"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=sorted(BENCHES), default="v3")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args(argv)
    from hackbarca.models.seq import predict_seq, train_seq

    names = BENCHES[args.bench]
    data = [game_inputs(g, i) for i, g in enumerate(names)]
    oof, gts = [], []
    for g in range(len(names)):
        train = [d[:5] for o, d in enumerate(data) if o != g]
        model = train_seq(train, epochs=args.epochs)
        tokens, mask, glob, _, _, n = data[g]
        p = predict_seq(model, tokens, mask, glob, n)
        oof.append(p)
        gts.append(load_bench_game(names[g])[1])
        acc = np.mean(np.argmax(p, 1)[gts[-1] != 3] == gts[-1][gts[-1] != 3])
        print("retenido {}: acierto por frame (argmax) {:.3f}".format(names[g], acc), flush=True)
    out = os.path.join(_REPO_ROOT, "external_data", "oof_seq_{}.npz".format(args.bench))
    np.savez_compressed(out, **{"p%d" % i: p for i, p in enumerate(oof)}, **{"gt%d" % i: g for i, g in enumerate(gts)})
    print("guardado", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
