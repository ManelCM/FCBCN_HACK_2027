"""U7: pose de los jugadores en el video tactico (saque de banda, portero con el balon), E091.

    scripts/run_aggregate.sh pose_pass uv run --group vision python -m hackbarca.vision.pose_pass

YOLO-pose (COCO, 17 puntos) sobre la misma rejilla que las personas de video_pass (1 de cada 5 frames
puntuados), por mosaicos. Por persona guarda, en data/derived/pose_pass/ (el agente no lo lee):
  frame, x, y (pie proyectado al campo, m), confianza,
  brazos arriba (las dos munecas por encima de la nariz, con confianza >= 0,3),
  manos juntas (distancia entre munecas / altura de la caja; NaN sin munecas),
  altura de las munecas (sobre la cadera, en alturas de caja; NaN sin puntos).
Un saque de banda es un jugador con los brazos arriba junto a la banda; un portero que tiene el balon en
las manos, manos juntas a la altura del pecho dentro del area. Solo imprime progreso y recuentos.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from hackbarca.vision.ball_detector import TILE, tiles, to_pitch
from hackbarca.vision.video_pass import BLOCK, PERSON_EVERY

NOSE, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 9, 10, 11, 12
KP_CONF = 0.3


def pose_rows(model, image, conf: float = 0.3, imgsz: int = TILE) -> np.ndarray:
    """(k, 6) en pixeles: pie x, y, confianza, brazos arriba, manos juntas, altura de las munecas."""
    corners = tiles(image.shape[1], image.shape[0])
    crops = [image[y:y + TILE, x:x + TILE] for x, y in corners]
    results = model.predict(crops, imgsz=imgsz, conf=conf, verbose=False)
    rows, seen = [], []
    for (x0, y0), res in zip(corners, results):
        if res.keypoints is None or not len(res.boxes):
            continue
        kp = res.keypoints.data.cpu().numpy()            # (n, 17, 3): x, y, confianza
        for (x1, y1, x2, y2), score, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), kp):
            fx, fy = x0 + (x1 + x2) / 2, y0 + y2
            if any(abs(fx - sx) < 8 and abs(fy - sy) < 8 for sx, sy in seen):
                continue
            seen.append((fx, fy))
            h = max(y2 - y1, 1.0)
            ok = k[:, 2] >= KP_CONF
            up = float(ok[NOSE] and ok[L_WRIST] and ok[R_WRIST] and k[L_WRIST, 1] < k[NOSE, 1]
                       and k[R_WRIST, 1] < k[NOSE, 1])
            together = np.nan
            if ok[L_WRIST] and ok[R_WRIST]:
                together = float(np.hypot(*(k[L_WRIST, :2] - k[R_WRIST, :2])) / h)
            height = np.nan
            if (ok[L_WRIST] or ok[R_WRIST]) and (ok[L_HIP] or ok[R_HIP]):
                wy = np.mean([k[j, 1] for j in (L_WRIST, R_WRIST) if ok[j]])
                hy = np.mean([k[j, 1] for j in (L_HIP, R_HIP) if ok[j]])
                height = float((hy - wy) / h)
            rows.append([fx, fy, float(score), up, together, height])
    return np.array(rows, dtype=float).reshape(-1, 6)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8s-pose.pt")
    parser.add_argument("--limit", type=int, default=0, help="prueba: solo N frames de la rejilla")
    parser.add_argument("--offset", type=int, default=0, help="prueba: empezar en este frame de la rejilla")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=2 * TILE, help="lado al que se escala cada mosaico (x2: jugadores pequenos)")
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
    wanted = wanted[wanted % PERSON_EVERY == 0]
    if args.limit:
        wanted = wanted[args.offset:args.offset + args.limit]
    out_dir = os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "derived",
                           "pose_pass" if not args.limit else "pose_pass_prueba")
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(args.weights)
    video = cv2.VideoCapture(require(["video"])[0])
    started = time.time()
    step = BLOCK // PERSON_EVERY
    blocks = [wanted[i:i + step] for i in range(0, len(wanted), step)]
    for b, block in enumerate(blocks):
        path = os.path.join(out_dir, "block%03d.npz" % b)
        if os.path.exists(path):
            continue
        rows = []
        video.set(cv2.CAP_PROP_POS_FRAMES, int(block[0]))
        frame, block_set = int(block[0]), set(block.tolist())
        while frame <= int(block[-1]):
            ok = video.grab()
            if not ok:
                break
            if frame in block_set and valid[frame]:
                _, image = video.retrieve()
                p = pose_rows(model, image, args.conf, args.imgsz)
                if len(p):
                    proj = to_pitch(p[:, :3], matrices[frame], pitch, margin=1e9)
                    if len(proj) == len(p):
                        inside = ((proj[:, 0] > -5) & (proj[:, 0] < pitch[0] + 5) & (proj[:, 1] > -5)
                                  & (proj[:, 1] < pitch[1] + 5))
                        for (x, y, s), extra in zip(proj[inside], p[inside, 3:]):
                            rows.append([frame, x, y, s, *extra])
            frame += 1
        np.savez_compressed(path, pose=np.array(rows, dtype=np.float32).reshape(-1, 7))
        arr = np.array(rows, dtype=float).reshape(-1, 7)
        print("bloque {}/{} ({:.0f} s): {} personas en {} frames | con munecas {:.2f} | brazos arriba {:.3f}".format(
            b + 1, len(blocks), time.time() - started, len(rows), len(block), np.mean(np.isfinite(arr[:, 5])) if len(arr) else 0,
            np.mean(arr[:, 4]) if len(arr) else 0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
