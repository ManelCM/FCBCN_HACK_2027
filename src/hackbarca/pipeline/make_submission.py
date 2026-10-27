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


def video_ball_path(data_dir: str | None) -> str:
    return os.path.join(data_dir or os.environ.get("MTAH_DATA_DIR", "data"), "derived", "video_ball.npz")


def predict(model: str, data_dir: str | None, dump_proba: str | None = None,
            short_dead: int | None = None) -> np.ndarray:
    """dump_proba: ruta (en data/derived/) donde guardar P(H/A/D) por frame y si el ATD ve el balon,
    para los agregados de `real_vs_bench` (ADR-0004: se queda en data/, no lo lee ningun agente)."""
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
        if config.get("video_fill"):
            # E063: balon del detector de video en los huecos del ATD (vision.ball_detector)
            from hackbarca.io.video_fill import apply_video_fill
            match = apply_video_fill(match, video_ball_path(data_dir))
        # descanso entre las dos mitades evaluadas: los equipos cambian de campo
        halftime = (GRADED[0][1] + GRADED[1][0]) // 2
        df = features_for(match, windows, halftime, config.get("features", 1))
        if config.get("extras"):
            # T2: familias de entrenador (features/coach.py), sobre el ATD real
            import pandas as pd

            from hackbarca.features.coach import extra_features
            from hackbarca.models.l3 import ensure_period
            df = pd.concat([df, extra_features(ensure_period(match, halftime), config["extras"])], axis=1)
        if config.get("broadcast"):
            # E057: jugadores visibles en el broadcast real en el mismo instante (alineacion E023/E026)
            import pandas as pd

            from hackbarca.io.broadcast_signal import broadcast_features
            from hackbarca.pipeline.real_signals import broadcast_players
            df = pd.concat([df, broadcast_features(broadcast_players(match.n_frames))], axis=1)
        if config.get("video_conductor"):
            # E075: quien conduce segun la pasada de video (vision/video_pass, data/derived/video_pass_full)
            import pandas as pd

            from hackbarca.io.video_conductor import conductor_features, real_conductor
            df = pd.concat([df, conductor_features(real_conductor(match))], axis=1)
        if config.get("stack"):
            proba = predict_stacked((load_model(), load_model(STAGE2_PATH)), df)
        else:
            proba = predict_proba(load_model(), df)
        if config.get("deep_weight"):
            # E034: promedio con el modelo profundo; prediccion en numpy con los pesos versionados
            # (la entrega no necesita torch ni GPU)
            from hackbarca.models.seq import load_numpy_state, predict_seq_numpy, seq_inputs
            from hackbarca.pipeline.train_l3_final import SEQ_PATH
            tokens, mask, glob = seq_inputs(match)
            w = config["deep_weight"]
            deep = predict_seq_numpy(load_numpy_state(SEQ_PATH), tokens, mask, glob, match.n_frames)
            proba = (1 - w) * proba + w * deep
        if dump_proba:
            os.makedirs(os.path.dirname(dump_proba), exist_ok=True)
            np.savez_compressed(dump_proba, proba=proba.astype(np.float32), ball_known=~np.isnan(match.ball_x))
            # y las features de la capa 1, para atribuir los errores (short_dead_report)
            np.savez_compressed(dump_proba.replace(".npz", "_features.npz"), columns=np.array(df.columns, dtype=str),
                                values=df.to_numpy(np.float32))
        params = DecodeParams(**config["decode"])
        p_change = None
        if params.alpha_switch or params.alpha_dead:
            from hackbarca.models.change import predict_change
            from hackbarca.pipeline.train_l3_final import CHANGE_PATHS
            p_change = predict_change(tuple(load_model(p) for p in CHANGE_PATHS), df)
        codes = decode(proba, list(GRADED), params, p_change)
        k = config.get("short_dead_frames", 0) if short_dead is None else short_dead
        if k:
            # E090: D de menos de k frames entre dos tramos vivos -> tramo anterior (failure_analysis)
            from hackbarca.pipeline.failure_analysis import drop_short_dead
            codes = drop_short_dead(codes, list(GRADED), k)
        return codes
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
    parser.add_argument("--short-dead", type=int, default=None,
                        help="E090: candidato con la regla de D cortos de k frames (por defecto, la del artefacto)")
    parser.add_argument("--dump-proba", default=None,
                        help="guarda las probabilidades (solo l3) en esta ruta de data/derived/; no cambia el CSV")
    args = parser.parse_args(argv)

    codes = predict(args.model, args.data_dir, args.dump_proba, args.short_dead)
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
