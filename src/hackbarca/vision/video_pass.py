"""Pasada completa del video tactico por GPU (U3 + U5 del plan completo, E074-E075).

    scripts/run_aggregate.sh video_pass uv run --group vision python -m hackbarca.vision.video_pass

Hasta ahora el detector de balon solo miraba los huecos del ATD (E063). Esta pasada mira TODOS los
frames puntuados y guarda, en data/derived/video_pass.npz (el agente no lo lee):
  - balon (U3): hasta 3 candidatos por frame (x, y en metros y confianza) con el YOLO afinado con
    pseudoetiquetas (E061). Sin filtrar: es el balon a 25 fps reales, con las patadas que el ATD
    aplana (E013), para buscar toques como Link y Hoernig 2017 (pico de aceleracion >= 4 m/s^2);
  - jugadores (U5), 1 de cada PERSON_EVERY frames: pie proyectado al campo (x, y), confianza y color
    medio del torso en Lab, para asignar equipo (y portero y arbitro) por color sin etiquetas.
Solo imprime progreso y recuentos. Se puede retomar: guarda por bloques en data/derived/video_pass_full/ (las pruebas con --limit, en video_pass/).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from hackbarca.vision.ball_detector import TILE, detect_frame, tiles, to_pitch

PERSON = 0             # clase COCO
PERSON_EVERY = 5       # frames
BALL_WEIGHTS = os.path.join("data", "derived", "ball_yolo", "runs", "atd_pl", "weights", "best.pt")
BLOCK = 5000


def detect_people(model, image, conf: float = 0.3) -> np.ndarray:
    """(k, 7) en pixeles: pie x, y, confianza y color Lab medio del torso."""
    import cv2
    corners = tiles(image.shape[1], image.shape[0])
    crops = [image[y:y + TILE, x:x + TILE] for x, y in corners]
    results = model.predict(crops, imgsz=TILE, conf=conf, classes=[PERSON], verbose=False)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    rows, seen = [], []
    for (x0, y0), res in zip(corners, results):
        for (x1, y1, x2, y2), score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            fx, fy = x0 + (x1 + x2) / 2, y0 + y2
            if any(abs(fx - sx) < 8 and abs(fy - sy) < 8 for sx, sy in seen):   # duplicado en el solape
                continue
            seen.append((fx, fy))
            h = y2 - y1
            ty1, ty2 = int(y0 + y1 + 0.2 * h), int(y0 + y1 + 0.5 * h)          # torso
            tx1, tx2 = int(x0 + x1 + 0.25 * (x2 - x1)), int(x0 + x2 - 0.25 * (x2 - x1))
            patch = lab[ty1:max(ty2, ty1 + 1), tx1:max(tx2, tx1 + 1)].reshape(-1, 3)
            color = patch.mean(0) if len(patch) else np.full(3, np.nan)
            rows.append([fx, fy, float(score), *color])
    return np.array(rows, dtype=float).reshape(-1, 6)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball-weights", default=BALL_WEIGHTS)
    parser.add_argument("--person-weights", default="yolov8s.pt")
    parser.add_argument("--ball-conf", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0, help="prueba: solo los primeros N frames puntuados")
    parser.add_argument("--out-dir", default=None,
                        help="carpeta de salida dentro de data/derived (por defecto video_pass_full o video_pass)")
    args = parser.parse_args(argv)

    import cv2
    from ultralytics import YOLO

    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED
    starter = os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "more-than-a-hack-2027", "starter")
    sys.path.insert(0, os.path.abspath(starter))
    import load_homography
    from games import require

    match = load_match()
    pitch = (match.pitch_length, match.pitch_width)
    matrices, _, valid = load_homography.load()
    wanted = np.concatenate([np.arange(a, b) for a, b in GRADED])
    if args.limit:
        wanted = wanted[:args.limit]
    out_dir = os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "derived",
                           args.out_dir or ("video_pass_full" if not args.limit else "video_pass"))
    os.makedirs(out_dir, exist_ok=True)
    ball_model, person_model = YOLO(args.ball_weights), YOLO(args.person_weights)
    video = cv2.VideoCapture(require(["video"])[0])
    started, frame = time.time(), 0
    blocks = [wanted[i:i + BLOCK] for i in range(0, len(wanted), BLOCK)]
    for k, block in enumerate(blocks):
        path = os.path.join(out_dir, "block%03d.npz" % k)
        if os.path.exists(path):
            continue
        video.set(cv2.CAP_PROP_POS_FRAMES, int(block[0]))
        frame = int(block[0])
        ball_rows, people_rows = [], []
        block_set = set(block.tolist())
        while frame <= int(block[-1]):
            ok, image = video.read()
            if not ok:
                break
            if frame in block_set and valid[frame]:
                c = to_pitch(detect_frame(ball_model, image, args.ball_conf, 0), matrices[frame], pitch)
                for row in c[np.argsort(-c[:, 2])][:3]:
                    ball_rows.append([frame, *row])
                if frame % PERSON_EVERY == 0:
                    p = detect_people(person_model, image)
                    if len(p):
                        # sin descartar en to_pitch (margen enorme) para no perder la pareja con el
                        # color; luego se queda lo que cae a < 5 m del campo
                        proj = to_pitch(p[:, :3], matrices[frame], pitch, margin=1e9)
                        if len(proj) == len(p):
                            inside = ((proj[:, 0] > -5) & (proj[:, 0] < pitch[0] + 5) & (proj[:, 1] > -5)
                                      & (proj[:, 1] < pitch[1] + 5))
                            for (x, y, s), col in zip(proj[inside], p[inside, 3:]):
                                people_rows.append([frame, x, y, s, *col])
            frame += 1
        np.savez_compressed(path, ball=np.array(ball_rows, dtype=np.float32).reshape(-1, 4),
                            people=np.array(people_rows, dtype=np.float32).reshape(-1, 7))
        print("bloque {}/{} ({:.0f} s): {} candidatos de balon, {} jugadores".format(
            k + 1, len(blocks), time.time() - started, len(ball_rows), len(people_rows)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
