"""U10: acciones con el balon y su equipo en el broadcast (T-DEED de SoccerNet Team Ball 2025), E097.

    scripts/run_aggregate.sh ball_actions uv run --group vision python -m hackbarca.vision.ball_action_pass

Modelo preentrenado sin reentrenar: el baseline oficial del reto SoccerNet Team Ball Action Spotting 2025
(T-DEED, RegNetY-002 + SGP-Mixer, GPL-3.0), clonado en external_data/tdeed/team y con los pesos de su Drive
en external_data/tdeed/ckpt (fuera de git; ver docs/investigacion/resultados/u10_ball_actions.md).
Se entreno con broadcast de television a 25 fps reducido a 796x448 y tomando 1 de cada 2 frames: aqui se
le da lo mismo, solo en las partes puntuadas (frames del broadcast segun la alineacion de E023).

Guarda en data/derived/ball_actions/half<k>.npz (el agente no lo lee): el frame del broadcast de cada
paso y las 25 probabilidades (fondo + 12 acciones x {equipo izquierda, derecha}). Solo imprime progreso
y recuentos.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

CLASSES = ("PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
           "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "external_data", "tdeed")
CLIP, STEP, STRIDE = 100, 50, 2
SIZE = (796, 448)


def load_model(device: str = "cuda"):
    import torch
    sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "team")))
    from model.model import TDEEDModel
    state = torch.load(os.path.join(ROOT, "ckpt", "SoccerNetTeamBall", "SoccerNetBall_baseline", "checkpoint_best.pt"),
                       map_location=device, weights_only=True)   # solo tensores: nada de pickle arbitrario
    args = SimpleNamespace(modality="rgb", temporal_arch="ed_sgp_mixer", radi_displacement=4,
                           feature_arch="rny002_gsf", event_team=True, clip_len=CLIP, n_layers=2, sgp_ks=9,
                           sgp_r=4, num_classes=len(CLASSES), crop_dim=None)
    model = TDEEDModel(device=device, args=args)
    if "_pred_fine._fc1._fc_out.weight" in state:   # dos cabezas (entrenado junto con SoccerNet)
        model._model.update_pred_head([state["_pred_fine._fc%d._fc_out.weight" % i].shape[0] for i in (1, 2)])
    model.load(state)
    return model


def run_clip(model, frames: list) -> np.ndarray:
    seq = np.stack(frames).transpose(0, 3, 1, 2)          # (L, C, H, W), RGB uint8
    _, proba = model.predict(seq)
    return proba[0]                                       # (L, 25)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="prueba: solo N pasos del modelo por parte")
    args = parser.parse_args(argv)

    import cv2

    from hackbarca.audio.whistle import half_offsets
    from hackbarca.pipeline.atd_stats import GRADED
    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    alignment = np.load(os.path.join(data_dir, "derived", "broadcast_alignment.npz"))
    halves = half_offsets(alignment["tactical_to_broadcast"], GRADED)
    out_dir = os.path.join(data_dir, "derived", "ball_actions" if not args.limit else "ball_actions_prueba")
    os.makedirs(out_dir, exist_ok=True)
    model = load_model()
    video = cv2.VideoCapture(os.path.join(data_dir, "broadcast", "BAR-ATH_broadcast.mp4"))
    started = time.time()
    for k, (lo, hi, _) in enumerate(halves):
        path = os.path.join(out_dir, "half%d.npz" % k)
        if os.path.exists(path):
            continue
        lo = max(lo, 0)
        video.set(cv2.CAP_PROP_POS_FRAMES, lo)
        frames, index, proba = [], [], []
        frame = lo
        while frame < hi and (not args.limit or len(index) < args.limit):
            ok = video.grab()
            if not ok:
                break
            if (frame - lo) % STRIDE == 0:
                _, image = video.retrieve()
                frames.append(cv2.cvtColor(cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                index.append(frame)
                if len(frames) == CLIP:
                    p = run_clip(model, frames)
                    # del primer clip se guarda tambien el principio; de cada clip, su mitad central
                    proba.append(p[:CLIP - (CLIP - STEP) // 2] if not proba else p[(CLIP - STEP) // 2:CLIP - (CLIP - STEP) // 2])
                    frames = frames[STEP:]
            frame += 1
        have = sum(len(p) for p in proba)
        if len(index) > have:   # cola: rellena hasta un clip y guarda lo que falta
            pad = frames + [frames[-1]] * (CLIP - len(frames))
            p = run_clip(model, pad)
            proba.append(p[len(frames) - (len(index) - have):len(frames)])
        proba = np.concatenate(proba)[:len(index)].astype(np.float16)
        index = np.array(index[:len(proba)], dtype=np.int64)
        np.savez_compressed(path, frame=index, proba=proba)
        events = (proba[:, 1:].max(1) >= 0.2).mean()
        print("parte {}: {} pasos en {:.0f} s | pasos con alguna accion >= 0,2: {:.3f}".format(
            k + 1, len(proba), time.time() - started, events), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
