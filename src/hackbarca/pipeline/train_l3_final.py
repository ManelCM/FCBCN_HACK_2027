"""Entrena el L3 final con todo el banco y guarda el artefacto versionado que usa la entrega.

    uv run python -m hackbarca.pipeline.train_l3_final --switch 20 --to-dead 10 --from-dead 10 --temperature 1

Los parametros de decodificacion se pasan explicitos (los elige `tune_l3` en el banco, nunca
con datos del reto). El modelo se versiona en git: la entrega no depende de reentrenar igual.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from hackbarca.models.l3 import DecodeParams, train
from hackbarca.pipeline.tune_l3 import load_bench

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "l3_lgbm.txt")
DECODE_PATH = os.path.join(ARTIFACTS, "l3_decode.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--switch", type=float, required=True)
    parser.add_argument("--to-dead", type=float, required=True)
    parser.add_argument("--from-dead", type=float, required=True)
    parser.add_argument("--no-tags", action="store_true")
    args = parser.parse_args(argv)

    games = load_bench(use_tags=not args.no_tags)
    model = train([df for df, _ in games], [gt for _, gt in games])
    os.makedirs(ARTIFACTS, exist_ok=True)
    model.save_model(MODEL_PATH)
    decode = DecodeParams(args.temperature, args.switch, args.to_dead, args.from_dead)
    with open(DECODE_PATH, "w") as handle:
        json.dump({"decode": asdict(decode), "use_tags": not args.no_tags}, handle, indent=2)
    print("guardado", MODEL_PATH, DECODE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
