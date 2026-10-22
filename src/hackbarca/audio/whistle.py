"""Detector de silbato en el audio del broadcast, llevado al timeline tactico (ADR-0013, E028).

El video tactico trae una pista de audio muda; el broadcast si tiene sonido y su grabacion es
continua dentro de cada parte (E026), asi que un desfase por parte lleva todo su audio al
timeline de la entrega. El silbato es un tono casi puro de ~2-4,5 kHz que dura unas decimas:
se busca, en cada frame de video (40 ms), el pico espectral de esa banda frente a la mediana de
la banda (tonalidad) y se exige que dure varios frames.

ADR-0004: lee el video del reto, asi que solo imprime agregados (`whistle_report`). Necesita el
grupo `audio` (imageio-ffmpeg, binario de ffmpeg fijado):

    uv sync --group audio
    scripts/run_aggregate.sh silbato uv run --group audio python -m hackbarca.audio.whistle
"""

from __future__ import annotations

import subprocess

import numpy as np

RATE = 16000            # Hz; el silbato vive en 2-4,5 kHz
FPS = 25                # frames de video: una columna de STFT por frame
HOP = RATE // FPS       # 640 muestras = 40 ms
NFFT = 1024
BAND = (2000.0, 4500.0)
MIN_FRAMES = 3          # >= 0,12 s seguidos por encima del umbral
Z_THRESHOLD = 6.0       # en desviaciones robustas (MAD) sobre la mediana del partido


def read_audio(path: str, rate: int = RATE) -> np.ndarray:
    """Audio mono en float32 a `rate` Hz, decodificado con el ffmpeg fijado en imageio-ffmpeg."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(rate), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def tonal_score(audio: np.ndarray, rate: int = RATE) -> np.ndarray:
    """Por frame de video: pico de la banda del silbato frente a su mediana, en dB."""
    n_frames = len(audio) // HOP
    window = np.hanning(NFFT).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / rate)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    pad = np.concatenate([np.zeros(NFFT // 2, np.float32), audio, np.zeros(NFFT, np.float32)])
    score = np.empty(n_frames, dtype=np.float32)
    chunk = 4096
    for a in range(0, n_frames, chunk):
        idx = np.arange(a, min(a + chunk, n_frames))[:, None] * HOP + np.arange(NFFT)[None, :]
        spec = np.abs(np.fft.rfft(pad[idx] * window, axis=1))[:, band] + 1e-9
        score[a:a + len(idx)] = 20 * np.log10(spec.max(1) / np.median(spec, axis=1))
    return score


def detect(score: np.ndarray, z: float = Z_THRESHOLD, min_frames: int = MIN_FRAMES):
    """[(inicio, fin)] en frames del broadcast donde la tonalidad supera el umbral robusto."""
    med = np.median(score)
    mad = np.median(np.abs(score - med)) * 1.4826 + 1e-9
    hot = (score - med) / mad > z
    edges = np.diff(np.concatenate(([0], hot.astype(np.int8), [0])))
    return [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a >= min_frames]


def half_offsets(tactical_to_broadcast: np.ndarray, halves) -> list[tuple[int, int, int]]:
    """[(primer_frame_broadcast, ultimo, desfase)] por parte: desfase = tactico - broadcast (moda)."""
    out = []
    for start, stop in halves:
        t = np.arange(start, stop)
        m = tactical_to_broadcast[start:stop]
        ok = m >= 0
        if not ok.any():
            continue
        values, counts = np.unique(t[ok] - m[ok], return_counts=True)
        offset = int(values[np.argmax(counts)])
        out.append((start - offset, stop - offset, offset))
    return out


def to_tactical(events, offsets) -> list[tuple[int, int]]:
    """Eventos del broadcast -> frames tacticos, solo los que caen dentro de una parte."""
    out = []
    for a, b in events:
        for lo, hi, offset in offsets:
            if lo <= a < hi:
                out.append((a + offset, b + offset))
                break
    return out


def whistle_report(whistles, gt=None, speed=None, graded=None) -> dict:
    """Agregados para decidir si el silbato sirve (ADR-0004: nada por frame)."""
    rep = {"silbatos": len(whistles)}
    if gt is not None:
        # transiciones vivo -> muerto de la GT publica y silbatos cerca (de -1 s a +2 s)
        known = gt != 3
        to_dead = [i for i in range(1, len(gt)) if known[i] and known[i - 1] and gt[i] == 2 and gt[i - 1] != 2]
        starts = np.array([a for a, _ in whistles])
        hit = [np.any((starts >= f - 25) & (starts <= f + 50)) for f in to_dead]
        window = np.flatnonzero(known)
        inside = starts[(starts >= window.min()) & (starts <= window.max())] if len(window) else starts[:0]
        near = [np.any([(s >= f - 25) and (s <= f + 50) for f in to_dead]) for s in inside]
        rep.update(gt_paso_a_D=len(to_dead), recall_paso_a_D=float(np.mean(hit)) if hit else float("nan"),
                   silbatos_en_ventana=len(inside),
                   precision_en_ventana=float(np.mean(near)) if near else float("nan"))
    if speed is not None and len(whistles):
        # velocidad colectiva 2-6 s despues frente a 2-6 s antes de cada silbato (parones: baja)
        drops = []
        for a, _ in whistles:
            if a - 150 >= 0 and a + 150 < len(speed):
                before, after = np.nanmean(speed[a - 150:a - 50]), np.nanmean(speed[a + 50:a + 150])
                if np.isfinite(before) and np.isfinite(after):
                    drops.append(after - before)
        rep["cambio_velocidad_mediano_m_s"] = float(np.median(drops)) if drops else float("nan")
        rep["frac_silbatos_con_frenada"] = float(np.mean(np.array(drops) < -0.3)) if drops else float("nan")
    if graded is not None:
        rep["silbatos_en_partes"] = int(sum(graded[a] for a, _ in whistles if a < len(graded)))
    return rep


def main(argv=None) -> int:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--z", type=float, default=Z_THRESHOLD)
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    args = parser.parse_args(argv)

    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED, PUBLIC_GT, collective_speed

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    video = os.path.join(data_dir, "broadcast", "BAR-ATH_broadcast.mp4")
    alignment = np.load(os.path.join(data_dir, "derived", "broadcast_alignment.npz"))
    offsets = half_offsets(alignment["tactical_to_broadcast"], GRADED)
    print("desfases por parte (tactico - broadcast, frames):", [o for _, _, o in offsets])

    score = tonal_score(read_audio(video))
    whistles = to_tactical(detect(score, args.z, args.min_frames), offsets)

    sys.path.insert(0, os.path.dirname(PUBLIC_GT))
    import possession_metrics as pm
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    match = load_match()
    graded = np.zeros(match.n_frames, bool)
    for a, b in GRADED:
        graded[a:b] = True
    report = whistle_report(whistles, gt=gt, speed=collective_speed(match), graded=graded)
    for key, value in report.items():
        print("{:<32} {}".format(key, round(value, 3) if isinstance(value, float) else value))
    out = os.path.join(data_dir, "derived", "whistles.npz")
    np.savez_compressed(out, start=np.array([a for a, _ in whistles]), stop=np.array([b for _, b in whistles]))
    print("guardado", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
