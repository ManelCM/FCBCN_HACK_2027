"""Entrena el L3 final con todo el banco y guarda el artefacto versionado que usa la entrega.

    uv run python -m hackbarca.pipeline.train_l3_final --switch 20 --to-dead 10 --from-dead 10 --temperature 1

Los parametros de decodificacion se pasan explicitos (los elige `tune_l3` en el banco, nunca
con datos del reto). El modelo se versiona en git: la entrega no depende de reentrenar igual.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import asdict

from hackbarca.models.l3 import DecodeParams, train, train_stacked
from hackbarca.pipeline.tune_l3 import BENCH_PARAMS, BENCHES, feature_spec, load_bench

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "l3_lgbm.txt.gz")
STAGE2_PATH = os.path.join(ARTIFACTS, "l3_stage2.txt.gz")  # solo si el modelo esta apilado (E018)
SEQ_PATH = os.path.join(ARTIFACTS, "seq_state.npz")       # modelo profundo (E034)
CHANGE_PATHS = (os.path.join(ARTIFACTS, "change_team.txt.gz"), os.path.join(ARTIFACTS, "change_dead.txt.gz"))
BUNDLE = "hackbarca-bundle\n"   # cabecera de los modelos que no son un solo Booster (M5d, M5e)


def save_model(model, path: str = MODEL_PATH) -> None:
    # mtime=0 y LF: el .gz es identico byte a byte entre ejecuciones y sistemas
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        text = model.model_to_string().replace("\r\n", "\n").encode()
    else:
        from hackbarca.models.l3 import model_to_dict
        text = (BUNDLE + json.dumps(model_to_dict(model), sort_keys=True)).encode()
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(text)


def load_model(path: str = MODEL_PATH):
    import lightgbm as lgb
    with gzip.open(path, "rt", newline="") as handle:
        text = handle.read()
    if text.startswith(BUNDLE):
        from hackbarca.models.l3 import model_from_dict
        return model_from_dict(json.loads(text[len(BUNDLE):]))
    return lgb.Booster(model_str=text)
DECODE_PATH = os.path.join(ARTIFACTS, "l3_decode.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--switch", type=float, required=True)
    parser.add_argument("--to-dead", type=float, required=True)
    parser.add_argument("--from-dead", type=float, required=True)
    parser.add_argument("--no-tags", action="store_true")
    parser.add_argument("--features", type=feature_spec, default=1)
    parser.add_argument("--stack", action="store_true", help="dos capas con contexto temporal (E018)")
    parser.add_argument("--bench", choices=sorted(BENCHES), default="v2",
                        help="partidos de entrenamiento: v2 = 3 Metrica; v3 = + 7 DFL; v3_1 = v3 con el balon de E047")
    parser.add_argument("--deep-weight", type=float, default=0.0,
                        help="peso del modelo profundo en el promedio con LightGBM (E034); 0 = sin el")
    parser.add_argument("--alpha-switch", type=float, default=0.0, help="coste de cambio dinamico (E032)")
    parser.add_argument("--alpha-dead", type=float, default=0.0)
    parser.add_argument("--extras", default="", help="T2: familias de entrenador (features/coach.py)")
    parser.add_argument("--broadcast", action="store_true",
                        help="E057: familia broadcast (simulada en el banco; la entrega usa la senal real)")
    parser.add_argument("--video-conductor", action="store_true",
                        help="familia conductor segun el video, simulada en el banco; la entrega usa el video real (E075)")
    parser.add_argument("--short-dead", type=int, default=75,
                        help="regla de D cortos de la entrega en frames (ADR-0021, E090); 0 = sin regla")
    parser.add_argument("--bag", type=int, default=1, help="M5e: media de N modelos con desfases del submuestreo")
    parser.add_argument("--xgb-mix", action="store_true", help="M5d: media de LightGBM y XGBoost en cada modelo")
    parser.add_argument("--video-fill", action="store_true",
                        help="banco con el relleno de balon por video simulado; la entrega lo aplica (E063)")
    args = parser.parse_args(argv)

    from hackbarca.models import l3
    l3.BAGS, l3.XGB_MIX = args.bag, args.xgb_mix
    fill = None
    if args.video_fill:
        from hackbarca.io.video_fill import VIDEO_FILL
        fill = VIDEO_FILL
    broadcast = None
    if args.broadcast:
        from hackbarca.io.broadcast_signal import BROADCAST_SIGNAL
        broadcast = BROADCAST_SIGNAL
    conductor = None
    if args.video_conductor:
        from hackbarca.io.video_conductor import VIDEO_CONDUCTOR
        conductor = VIDEO_CONDUCTOR
    games = load_bench(use_tags=not args.no_tags, version=args.features, params=BENCH_PARAMS[args.bench],
                       names=BENCHES[args.bench], fill=fill, broadcast=broadcast,
                       extras=tuple(x for x in args.extras.split("+") if x), conductor=conductor)
    os.makedirs(ARTIFACTS, exist_ok=True)
    features, labels = [df for df, _ in games], [gt for _, gt in games]
    if args.stack:
        stage1, stage2 = train_stacked(features, labels)
        save_model(stage1)
        save_model(stage2, STAGE2_PATH)
    else:
        save_model(train(features, labels))
        if os.path.exists(STAGE2_PATH):
            os.remove(STAGE2_PATH)
    for path in (SEQ_PATH,) + CHANGE_PATHS:   # nada de restos de un artefacto anterior
        if os.path.exists(path):
            os.remove(path)
    if args.alpha_switch or args.alpha_dead:
        from hackbarca.models.change import train_change
        for model, path in zip(train_change(features, labels), CHANGE_PATHS):
            save_model(model, path)
    if args.deep_weight:
        from hackbarca.models.seq import save_state, train_seq
        from hackbarca.pipeline.tune_seq import game_inputs
        names = BENCHES[args.bench]
        save_state(train_seq([game_inputs(g, i)[:5] for i, g in enumerate(names)]), SEQ_PATH)
    decode = DecodeParams(args.temperature, args.switch, args.to_dead, args.from_dead,
                          args.alpha_switch, args.alpha_dead)
    with open(DECODE_PATH, "w") as handle:
        json.dump({"decode": asdict(decode), "use_tags": not args.no_tags, "features": args.features,
                   "stack": args.stack, "bench": args.bench, "deep_weight": args.deep_weight,
                   "video_fill": args.video_fill, "broadcast": args.broadcast,
                   "video_conductor": args.video_conductor, "short_dead_frames": args.short_dead,
                   "bag": args.bag, "xgb_mix": args.xgb_mix,
                   "extras": [x for x in args.extras.split("+") if x]},
                  handle, indent=2)
    print("guardado", MODEL_PATH, DECODE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
