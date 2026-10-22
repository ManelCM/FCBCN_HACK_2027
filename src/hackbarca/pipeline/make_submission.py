"""Un unico comando para regenerar submissions/<equipo>.csv con un modelo del repo.

ADR-0004: lee data/ (tracking real), asi que lo ejecuta una persona:

    uv run python -m hackbarca.pipeline.make_submission --model l1 --team-name <equipo>

Determinista: mismos datos + mismo commit -> mismo fichero byte a byte. Los parametros de cada
modelo estan congelados en el codigo (elegidos en el banco externo, nunca con datos del reto).
Fuera de las dos mitades evaluadas escribe X (no se puntua); dentro nunca escribe X.
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

from hackbarca.eval.evaluate import validate_csv
from hackbarca.pipeline.atd_stats import GRADED

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
LETTERS = np.array(["H", "A", "D"])


def predict(model: str, data_dir: str | None) -> np.ndarray:
    from hackbarca.io.load_match import load_match
    match = load_match(data_dir=data_dir)
    if model == "baseline":
        from hackbarca.models.baseline import predict_baseline
        return predict_baseline(match, 50)
    if model == "l1":
        from hackbarca.io.smart_tagging import set_piece_windows
        from hackbarca.models.l1 import predict_l1
        from hackbarca.models.l1_config import L1_FROZEN
        return predict_l1(match, list(GRADED), L1_FROZEN, set_piece_windows(data_dir=data_dir))
    if model == "l3":
        import json

        from hackbarca.io.smart_tagging import set_piece_windows
        from hackbarca.models.l3 import DecodeParams, decode, features_for, predict_proba
        from hackbarca.models.l3 import predict_stacked
        from hackbarca.pipeline.train_l3_final import DECODE_PATH, STAGE2_PATH, load_model
        with open(DECODE_PATH) as handle:
            config = json.load(handle)
        windows = set_piece_windows(data_dir=data_dir) if config["use_tags"] else None
        # descanso entre las dos mitades evaluadas: los equipos cambian de campo
        halftime = (GRADED[0][1] + GRADED[1][0]) // 2
        df = features_for(match, windows, halftime, config.get("features", 1))
        if config.get("stack"):
            proba = predict_stacked((load_model(), load_model(STAGE2_PATH)), df)
        else:
            proba = predict_proba(load_model(), df)
        return decode(proba, list(GRADED), DecodeParams(**config["decode"]))
    raise ValueError("modelo desconocido: " + model)


def write_csv(path: str, codes: np.ndarray) -> None:
    labels = np.full(len(codes), "X", dtype="<U1")
    for start, stop in GRADED:
        labels[start:stop] = LETTERS[codes[start:stop]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "l1", "l3"), required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "submissions"))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    codes = predict(args.model, args.data_dir)
    path = os.path.join(args.out_dir, "{}.csv".format(args.team_name))
    write_csv(path, codes)
    result = validate_csv(path)
    if not result.ok:
        print("VALIDACION FALLIDA: {}".format(result.reason))
        return 1
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()[:16]
    # Solo agregados (ADR-0004): recuentos, n de posesiones y huella del fichero.
    print("OK {} filas | posesiones={} duracion media={:.1f}s | labels={} | sha256={}".format(
        result.n_rows, result.n_spells, result.mean_spell_duration_s, result.label_counts, digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
