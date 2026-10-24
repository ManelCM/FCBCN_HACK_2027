"""Detector de balon afinado con pseudoetiquetas del propio ATD (plan §4.1, variante abierta de ADR-0010).

El YOLO generico (COCO) no ve un balon tan pequeno en el plano tactico: solo el 24 % a < 2 m (E025).
Pero en el 59 % de los frames el ATD si sabe donde esta el balon, y la homografia oficial da su pixel.
Con eso se construye un conjunto de entrenamiento etiquetado por un ALGORITMO (ninguna etiqueta manual,
RULES.md) y se afina YOLO. Los tramos que usa la prueba `ball_detector --sim-gaps 40` se excluyen (+-10 s)
para que la evaluacion sea honesta.

ADR-0004: lee el video del reto. Las imagenes se guardan en data/derived/ball_yolo/ (el agente no las
abre) y los scripts solo imprimen agregados.

    scripts/run_aggregate.sh yolo_datos   uv run --group vision python -m hackbarca.vision.pseudo_labels build
    scripts/run_aggregate.sh yolo_entreno uv run --group vision python -m hackbarca.vision.pseudo_labels train
    scripts/run_aggregate.sh yolo_prueba  uv run --group vision python -m hackbarca.vision.ball_detector \
        --weights data/derived/ball_yolo/runs/atd_pl/weights/best.pt --class-id 0 --sim-gaps 40
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

TILE = 640
BOX = 18          # px: lado de la caja del balon en el plano tactico (1080p)
N_POS = 6000      # recortes con balon
NEG_FRAC = 0.25   # recortes sin balon (zonas lejos del balon en frames con balon)
VAL_FRAC = 0.1
EXCLUDE = 250     # frames (+-10 s) alrededor de los tramos de la prueba --sim-gaps
SEED = 0


def out_dir() -> str:
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "derived", "ball_yolo")


def build() -> None:
    import cv2

    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED
    from hackbarca.vision.ball_detector import _sim_windows
    starter = os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "more-than-a-hack-2027", "starter")
    sys.path.insert(0, os.path.abspath(starter))
    import load_homography
    from games import require

    rng = np.random.default_rng(SEED)
    match = load_match()
    matrices, _, valid = load_homography.load()
    graded = np.zeros(match.n_frames, bool)
    for a, b in GRADED:
        graded[a:b] = True
    known = ~np.isnan(match.ball_x)
    eligible = graded & known & valid.astype(bool)
    for a, b in _sim_windows(graded & known, 40):
        eligible[max(0, a - EXCLUDE):b + EXCLUDE] = False
    cand = np.flatnonzero(eligible)
    picks = cand[np.unique(np.linspace(0, len(cand) - 1, N_POS).round().astype(int))]
    wanted = set(picks.tolist())

    root = out_dir()
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
    video = cv2.VideoCapture(require(["video"])[0])
    stats = dict(pos=0, neg=0, outside=0)
    frame, last = 0, int(picks.max())
    while frame <= last:
        if not video.grab():
            break
        if frame in wanted:
            _, image = video.retrieve()
            h, w = image.shape[:2]
            px, py = load_homography.project(matrices[frame], match.ball_x[frame] / match.pitch_length,
                                             match.ball_y[frame] / match.pitch_width)
            px, py = float(np.squeeze(px)), float(np.squeeze(py))
            if not (BOX <= px < w - BOX and BOX <= py < h - BOX):
                stats["outside"] += 1
            else:
                split = "val" if rng.random() < VAL_FRAC else "train"
                x0 = int(np.clip(px - rng.uniform(64, TILE - 64), 0, w - TILE))
                y0 = int(np.clip(py - rng.uniform(64, TILE - 64), 0, h - TILE))
                tile = image[y0:y0 + TILE, x0:x0 + TILE]
                name = "f%06d" % frame
                cv2.imwrite(os.path.join(root, "images", split, name + ".jpg"), tile)
                with open(os.path.join(root, "labels", split, name + ".txt"), "w") as handle:
                    handle.write("0 %.6f %.6f %.6f %.6f\n" % ((px - x0) / TILE, (py - y0) / TILE, BOX / TILE, BOX / TILE))
                stats["pos"] += 1
                if rng.random() < NEG_FRAC:   # recorte sin balon, lejos de el
                    for _ in range(10):
                        nx, ny = int(rng.uniform(0, w - TILE)), int(rng.uniform(0, h - TILE))
                        if not (nx - 2 * BOX <= px < nx + TILE + 2 * BOX and ny - 2 * BOX <= py < ny + TILE + 2 * BOX):
                            cv2.imwrite(os.path.join(root, "images", split, name + "_n.jpg"), image[ny:ny + TILE, nx:nx + TILE])
                            open(os.path.join(root, "labels", split, name + "_n.txt"), "w").close()
                            stats["neg"] += 1
                            break
        frame += 1
    with open(os.path.join(root, "data.yaml"), "w") as handle:
        handle.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n" % os.path.abspath(root).replace("\\", "/"))
    print("recortes con balon %d | sin balon %d | balon fuera de imagen (descartados) %d" % (stats["pos"], stats["neg"], stats["outside"]))


def train(epochs: int = 30) -> None:
    from ultralytics import YOLO
    root = out_dir()
    model = YOLO("yolov8n.pt")
    model.train(data=os.path.join(root, "data.yaml"), imgsz=TILE, epochs=epochs, batch=16, seed=SEED,
                deterministic=True, project=os.path.join(root, "runs"), name="atd_pl", exist_ok=True,
                workers=2, plots=False, verbose=False)
    metrics = model.val(data=os.path.join(root, "data.yaml"), imgsz=TILE, verbose=False, plots=False)
    print("validacion (recortes apartados): precision %.3f recall %.3f mAP50 %.3f" % (
        metrics.box.mp, metrics.box.mr, metrics.box.map50))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("build", "train"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    build() if args.step == "build" else train(args.epochs)
