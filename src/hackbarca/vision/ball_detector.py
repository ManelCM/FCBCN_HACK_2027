"""Detector de balon en el video tactico para rellenar los huecos del ATD (Frente A, paso 3).

ADR-0004: lee el video del reto, asi que lo ejecuta una persona EN LOCAL (GPU). Nada sale de la
maquina; solo se imprimen agregados. Pesos abiertos preentrenados (COCO "sports ball"); nada se
etiqueta a mano (RULES.md). Requiere el grupo de dependencias de vision:

    uv sync --group vision
    uv run python -m hackbarca.vision.ball_detector --weights yolov8s.pt

Salida: data/derived/video_ball.npz con frame, x, y (m) para los frames de hueco evaluados.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from hackbarca.vision.ball_track import link_candidates

SPORTS_BALL = 32  # clase COCO
TILE = 640
OVERLAP = 96


def tiles(width: int = 1920, height: int = 1080, size: int = TILE, overlap: int = OVERLAP):
    """Esquinas (x0, y0) de mosaicos size x size que cubren la imagen con solape."""
    def starts(total):
        step = size - overlap
        out = list(range(0, max(total - size, 0) + 1, step))
        if out[-1] + size < total:
            out.append(total - size)
        return out
    return [(x, y) for y in starts(height) for x in starts(width)]


def detect_frame(model, image, conf: float = 0.1, class_id: int = SPORTS_BALL) -> np.ndarray:
    """Candidatos (k, 3) en pixeles: x, y del centro y confianza."""
    corners = tiles(image.shape[1], image.shape[0])
    crops = [image[y:y + TILE, x:x + TILE] for x, y in corners]
    results = model.predict(crops, imgsz=TILE, conf=conf, classes=[class_id], verbose=False)
    rows = []
    for (x0, y0), res in zip(corners, results):
        for box, score in zip(res.boxes.xywh.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            rows.append([x0 + box[0], y0 + box[1], float(score)])
    return np.array(rows).reshape(-1, 3)


def to_pitch(cands_px: np.ndarray, matrix: np.ndarray, pitch, margin: float = 3.0) -> np.ndarray:
    """Pixeles -> metros con la homografia (inversa de la oficial); descarta fuera del campo."""
    if not len(cands_px) or np.isnan(matrix).any():
        return np.zeros((0, 3))
    inv = np.linalg.inv(matrix)
    px, py = cands_px[:, 0], cands_px[:, 1]
    den = inv[2, 0] * px + inv[2, 1] * py + inv[2, 2]
    x = (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]) / den * pitch[0]
    y = (inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]) / den * pitch[1]
    keep = (x > -margin) & (x < pitch[0] + margin) & (y > -margin) & (y < pitch[1] + margin)
    return np.column_stack([x, y, cands_px[:, 2]])[keep]


def _runs(frames: np.ndarray):
    if not len(frames):
        return []
    cuts = np.flatnonzero(np.diff(frames) != 1) + 1
    return np.split(frames, cuts)


SIM_GAP = 250  # frames (10 s), del orden de la mediana de los huecos largos del ATD (268)


def _sim_windows(ball_seen: np.ndarray, n: int, length: int = SIM_GAP):
    """N tramos [a, b) repartidos por el partido con el balon visible en el ATD en todo el tramo."""
    edges = np.diff(np.concatenate(([0], ball_seen.astype(np.int8), [0])))
    starts = [a for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
              if b - a >= length]
    if not starts:
        return []
    picks = np.unique(np.linspace(0, len(starts) - 1, n).round().astype(int))
    return [(starts[i], starts[i] + length) for i in picks]


def auto_conf_ref(cands: dict) -> float:
    """Mediana de la confianza maxima por frame (solo frames con candidatos): sale del detector, sin GT."""
    tops = [float(c[:, 2].max()) for c in cands.values() if len(c)]
    return float(np.median(tops)) if tops else 1.0


def _cache_path() -> str:
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "derived", "sim_gaps_cands.npz")


def _save_cands(cands: dict, path: str) -> None:
    frames = np.array(sorted(cands), dtype=int)
    rows = [cands[f] for f in frames]
    np.savez_compressed(path, frame=frames, count=np.array([len(r) for r in rows], dtype=int),
                        cand=np.concatenate(rows) if rows else np.zeros((0, 3)))


def _load_cands(path: str) -> dict:
    data = np.load(path)
    parts = np.split(data["cand"], np.cumsum(data["count"])[:-1])
    return {int(f): p.reshape(-1, 3) for f, p in zip(data["frame"], parts)}


def _report_sim_gaps(windows, cands, match, conf_ref: float = 1.0) -> None:
    """Agregados del detector en huecos simulados, con el balon del ATD como referencia."""
    top_err, any_err, path_err, covered = [], [], [], []
    for a, b in windows:
        frames = np.arange(a, b)
        ref = np.column_stack([match.ball_x[frames], match.ball_y[frames]])
        for f, (rx, ry) in zip(frames, ref):
            c = cands.get(int(f), np.zeros((0, 3)))
            d = np.hypot(c[:, 0] - rx, c[:, 1] - ry) if len(c) else np.array([np.inf])
            any_err.append(d.min())
            top_err.append(d[np.argmax(c[:, 2])] if len(c) else np.inf)
        path = link_candidates([cands.get(int(f), np.zeros((0, 3))) for f in frames], match.fps, conf_ref)
        ok = ~np.isnan(path[:, 0])
        covered.append(ok)
        path_err.append(np.where(ok, np.hypot(path[:, 0] - ref[:, 0], path[:, 1] - ref[:, 1]), np.inf))
    top_err, any_err = np.array(top_err), np.array(any_err)
    path_err, covered = np.concatenate(path_err), np.concatenate(covered)
    print("huecos simulados: {} tramos de {} frames (conf_ref {:.3f})".format(len(windows), SIM_GAP, conf_ref))
    print("  algun candidato a < 2 m: {:.3f}   el de mas confianza a < 2 m: {:.3f}".format(
        float(np.mean(any_err < 2)), float(np.mean(top_err < 2))))
    print("  tras enlazar: cobertura {:.3f}  a < 2 m (sobre todos los frames) {:.3f}  "
          "error mediano en los cubiertos {:.2f} m".format(
              float(covered.mean()), float(np.mean(path_err < 2)),
              float(np.median(path_err[covered])) if covered.any() else float("nan")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--class-id", type=int, default=SPORTS_BALL,
                        help="clase del detector: 32 (COCO sports ball) o 0 (modelo afinado con pseudoetiquetas)")
    parser.add_argument("--validate-every", type=int, default=25,
                        help="ademas, 1 de cada N frames con balon en el ATD para medir el acuerdo")
    parser.add_argument("--limit", type=int, default=0, help="procesar solo N frames (prueba)")
    parser.add_argument("--validate-only", action="store_true",
                        help="solo los frames de validacion, repartidos por todo el partido (sin huecos)")
    parser.add_argument("--sim-gaps", type=int, default=0,
                        help="diagnostico: N tramos de 10 s CON balon en el ATD tratados como huecos "
                             "(deteccion + enlace) y comparados con el ATD; guarda los candidatos en "
                             "data/derived/sim_gaps_cands.npz")
    parser.add_argument("--reuse-cands", action="store_true",
                        help="con --sim-gaps: reutiliza los candidatos guardados (solo repite el enlace)")
    parser.add_argument("--conf-ref", default="1",
                        help="confianza tratada como segura en el enlace: un numero o 'auto' "
                             "(mediana de la confianza maxima por frame del propio detector)")
    args = parser.parse_args(argv)

    import cv2
    from ultralytics import YOLO

    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED
    starter = os.path.join(os.path.dirname(__file__), "..", "..", "..", "external",
                           "more-than-a-hack-2027", "starter")
    sys.path.insert(0, os.path.abspath(starter))
    import load_homography
    from games import require

    match = load_match()
    pitch = (match.pitch_length, match.pitch_width)
    matrices, _, valid = load_homography.load()
    graded = np.zeros(match.n_frames, bool)
    for a, b in GRADED:
        graded[a:b] = True
    missing = np.isnan(match.ball_x)
    gap_frames = np.flatnonzero(graded & missing)
    check_frames = np.flatnonzero(graded & ~missing)[::args.validate_every]
    if args.validate_only:
        gap_frames = gap_frames[:0]
    windows = _sim_windows(graded & ~missing, args.sim_gaps) if args.sim_gaps else []
    if args.sim_gaps:
        gap_frames, check_frames = gap_frames[:0], check_frames[:0]
    wanted = np.union1d(gap_frames, check_frames)
    if windows:
        wanted = np.union1d(wanted, np.concatenate([np.arange(a, b) for a, b in windows]))
    if args.limit:
        wanted = wanted[:args.limit]
    wanted_set = set(wanted.tolist())

    started, frame = time.time(), 0
    if windows and args.reuse_cands:
        cands = _load_cands(_cache_path())
        conf_ref = auto_conf_ref(cands) if args.conf_ref == "auto" else float(args.conf_ref)
        _report_sim_gaps(windows, cands, match, conf_ref)
        return 0

    model = YOLO(args.weights)
    video = cv2.VideoCapture(require(["video"])[0])
    cands = {}
    last = int(wanted.max()) if len(wanted) else -1
    while frame <= last:
        ok = video.grab()
        if not ok:
            break
        if frame in wanted_set:
            _, image = video.retrieve()
            cands[frame] = to_pitch(detect_frame(model, image, args.conf, args.class_id), matrices[frame], pitch) \
                if valid[frame] else np.zeros((0, 3))
        frame += 1
        if frame % 10000 == 0:
            print("frame {} ({:.0f}s)".format(frame, time.time() - started), flush=True)

    conf_ref = auto_conf_ref(cands) if args.conf_ref == "auto" else float(args.conf_ref)
    if windows:
        os.makedirs(os.path.dirname(_cache_path()), exist_ok=True)
        _save_cands(cands, _cache_path())
        _report_sim_gaps(windows, cands, match, conf_ref)
        print("({:.0f}s)".format(time.time() - started))
        return 0

    # huecos: trayectoria enlazada por tramos consecutivos
    xs, ys, fs = [], [], []
    for run in _runs(np.array([f for f in gap_frames if f in cands])):
        path = link_candidates([cands[f] for f in run], match.fps, conf_ref)
        fs.append(run)
        xs.append(path[:, 0])
        ys.append(path[:, 1])
    fs = np.concatenate(fs) if fs else np.zeros(0, int)
    xs = np.concatenate(xs) if xs else np.zeros(0)
    ys = np.concatenate(ys) if ys else np.zeros(0)

    # validacion: candidato de mas confianza frente al balon del ATD (solo agregados)
    errs, seen = [], 0
    for f in check_frames:
        c = cands.get(int(f))
        if c is None:
            continue
        if len(c):
            seen += 1
            best = c[np.argmax(c[:, 2])]
            errs.append(np.hypot(best[0] - match.ball_x[f], best[1] - match.ball_y[f]))
    errs = np.array(errs)
    print("frames de hueco evaluados: {}  con balon tras enlazar: {:.3f}".format(
        len(fs), float(np.mean(~np.isnan(xs))) if len(xs) else 0.0))
    if len(errs):
        print("validacion ({} frames con balon ATD): deteccion {:.3f}  error mediano {:.2f} m  "
              "a < 2 m {:.3f}".format(len(check_frames), seen / max(len(check_frames), 1),
                                    float(np.median(errs)), float(np.mean(errs < 2))))
    out = os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), "derived", "video_ball.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, frame=fs, x=xs, y=ys)
    print("guardado", out, "({:.0f}s)".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
