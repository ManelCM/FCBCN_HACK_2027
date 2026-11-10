"""Silbato v2: detector de varios tonos paralelos y su puerta con pseudoetiquetas (L2, E083).

E029 buscaba un solo tono (pico frente a mediana de 2-4,5 kHz, columnas de 40 ms) y no encontro
nada. Un silbato de bola da 2-3 tonos planos y paralelos entre ~3,3 y 5,5 kHz durante 0,2-1,5 s;
el comentario vive por debajo de ~3,5 kHz y la grada es de banda ancha. Por columna de STFT
(bins de ~8 Hz, salto de 16 ms):

1. prominencia: dB sobre la mediana de +-160 Hz alrededor (estrecho en frecuencia);
2. novedad: dB sobre la mediana del mismo bin en +-2 s (tiene inicio; mata las lineas fijas de
   la mezcla, que con la tonalidad de E029 dan "silbato" todo el partido);
3. linea = min(1, 2) con una mediana de ~0,24 s en el tiempo (sostenido a un tono);
4. puntuacion = la segunda linea mas fuerte a >= 250 Hz de la primera (dos tonos a la vez).

Un silbato es una racha con puntuacion >= umbral de >= 0,15 s (rachas a < 0,3 s se unen). Su papel
sale del tracking, sin etiquetas: si la velocidad colectiva baja es una parada y si sube, una
reanudacion.

Puerta (docs/06_plan_parones.md, L2), solo agregados y pseudoetiquetas (prediccion segura); la GT
de la ventana publica no se usa (ADR-0019):

    uv sync --group audio
    scripts/run_aggregate.sh silbato_v2 uv run --group audio python -m hackbarca.audio.whistle_v2
"""

from __future__ import annotations

import json
import os

import numpy as np

RATE = 16000
NPERSEG = 2048          # bins de 7,8 Hz: un tono de silbato ocupa pocos bins
HOP = 256               # 16 ms
BAND = (3300.0, 5500.0)
PROMINENCE_HZ = 160.0
NOVELTY_S = 2.0
NOVELTY_STRIDE = 8      # la mediana temporal se calcula cada 8 columnas (~0,13 s) e interpola
HOLD_S = 0.24
SECOND_TONE_HZ = 250.0
SCORE_MIN = 12.0        # dB; pre-registrado antes de ver el reto
THRESHOLDS = (8.0, 10.0, 12.0, 15.0, 20.0)  # solo informativo
MERGE_S = 0.3
MIN_S = 0.15
CHUNK_S = 120
PAD_S = 6
FPS = 25
ROLE_BEFORE_S = (-3.0, 0.0)
ROLE_AFTER_S = (2.0, 6.0)   # salta los 2 s en que los jugadores aun reaccionan
ROLE_DELTA = 0.3            # m/s

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(_REPO_ROOT, "docs", "investigacion", "resultados", "silbato_v2.json")


def whistle_score(audio: np.ndarray, rate: int = RATE):
    """(tiempo en s, puntuacion en dB, tono principal en Hz) por columna de STFT."""
    from scipy.ndimage import maximum_filter1d, median_filter
    from scipy.signal import stft

    bin_hz = rate / NPERSEG
    prom_bins = 2 * int(round(PROMINENCE_HZ / bin_hz)) + 1
    hold_cols = max(1, int(round(HOLD_S * rate / HOP)))
    nov_cols = 2 * int(round(NOVELTY_S * rate / HOP / NOVELTY_STRIDE)) + 1
    chunk, pad = CHUNK_S * rate, PAD_S * rate
    times, scores, pitches = [], [], []
    for c0 in range(0, len(audio), chunk):
        a = max(0, c0 - pad)
        f, t, z = stft(audio[a:c0 + chunk + pad], rate, nperseg=NPERSEG, noverlap=NPERSEG - HOP,
                       boundary=None, padded=False)
        if z.shape[1] == 0:
            continue
        band = (f >= BAND[0] - 300) & (f <= BAND[1] + 300)  # margen para la mediana de los vecinos
        db = (20 * np.log10(np.abs(z[band]) + 1e-9)).astype(np.float32)
        prominence = db - median_filter(db, size=(prom_bins, 1))
        coarse = median_filter(db[:, ::NOVELTY_STRIDE], size=(1, nov_cols))
        reference = np.repeat(coarse, NOVELTY_STRIDE, axis=1)[:, :db.shape[1]]
        line = median_filter(np.minimum(prominence, db - reference), size=(1, hold_cols))

        fb = f[band]
        inside = (fb >= BAND[0]) & (fb <= BAND[1])
        line, freqs = line[inside], fb[inside]
        strongest = line.argmax(0)
        far = np.abs(freqs[:, None] - freqs[strongest][None, :]) >= SECOND_TONE_HZ
        second = np.where(far, maximum_filter1d(line, 3, axis=0), -np.inf).max(0)

        t = t + a / rate
        keep = (t >= c0 / rate) & (t < (c0 + chunk) / rate)
        times.append(t[keep])
        scores.append(second[keep])
        pitches.append(freqs[strongest][keep])
    return np.concatenate(times), np.concatenate(scores), np.concatenate(pitches)


def stationary_lines(audio: np.ndarray, rate: int = RATE, step_s: float = 10.0, min_db: float = 6.0):
    """Diagnostico de E029: bins de 2-5,5 kHz que sobresalen de sus vecinos casi todo el partido."""
    from scipy.ndimage import median_filter

    n = NPERSEG
    window = np.hanning(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    band = (freqs >= 2000) & (freqs <= 5500)
    starts = np.arange(0, len(audio) - n, int(step_s * rate))
    spec = np.stack([20 * np.log10(np.abs(np.fft.rfft(audio[s:s + n] * window))[band] + 1e-9) for s in starts])
    prom = spec - median_filter(spec, size=(1, 2 * int(round(PROMINENCE_HZ * n / rate)) + 1))
    share = (prom >= min_db).mean(0)  # fraccion del partido en que cada bin sobresale
    fb = freqs[band]
    persistent = share >= 0.5
    return dict(bins_fijos=int(persistent.sum()),
                hz_fijos=[int(round(h)) for h in fb[persistent][:20]],
                max_fraccion=float(share.max()), hz_max=int(round(fb[share.argmax()])))


def find_whistles(t, score, threshold: float = SCORE_MIN):
    """[(inicio_s, fin_s, pico_dB)] de cada silbato."""
    edges = np.diff(np.concatenate(([0], (score >= threshold).astype(np.int8), [0])))
    merged = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        if merged and t[s] - t[merged[-1][1]] < MERGE_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(float(t[s]), float(t[e]), float(score[s:e + 1].max())) for s, e in merged if t[e] - t[s] >= MIN_S]


def to_tactical_frames(whistles, offsets):
    """Silbatos (s del broadcast) -> frames tacticos de inicio, solo los que caen dentro de una parte."""
    from hackbarca.audio.whistle import to_tactical
    events = [(int(round(a * FPS)), int(round(b * FPS))) for a, b, _ in whistles]
    return np.array([a for a, _ in to_tactical(events, offsets)], dtype=np.int64)


def roles(starts: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """+1 parada (los jugadores frenan), -1 reanudacion (aceleran), 0 dudoso o sin tracking."""
    out = np.zeros(len(starts), dtype=np.int8)
    for i, f in enumerate(starts):
        b0, b1 = f + int(ROLE_BEFORE_S[0] * FPS), f + int(ROLE_BEFORE_S[1] * FPS)
        a0, a1 = f + int(ROLE_AFTER_S[0] * FPS), f + int(ROLE_AFTER_S[1] * FPS)
        if b0 < 0 or a1 > len(speed):
            continue
        before, after = speed[b0:b1], speed[a0:a1]
        if not (np.isfinite(before).any() and np.isfinite(after).any()):
            continue
        d = np.nanmean(after) - np.nanmean(before)
        out[i] = 1 if d < -ROLE_DELTA else (-1 if d > ROLE_DELTA else 0)
    return out


def reference_events(pred, pmax, graded_runs):
    """Pseudoetiquetas: inicios y finales de los D seguros (>= 5 s, media de pmax >= 0,9, entre vivos)
    y los D cortos predichos (< 5 s). Nada de GT."""
    from hackbarca.eval.evaluate import pm
    D = 2
    stops, restarts, short = [], [], []
    for a, b in graded_runs:
        segs = pm.segments(pred, [(a, b)])
        for k, (s, e, c) in enumerate(segs):
            if c != D or k == 0 or k == len(segs) - 1:
                continue
            if e - s >= 5 * FPS and pmax[s:e].mean() >= 0.9:
                stops.append(s)
                restarts.append(e)
            elif e - s < 5 * FPS:
                short.append((s, e))
    return np.array(stops), np.array(restarts), short


def _near(events: np.ndarray, refs: np.ndarray, lo_s: float, hi_s: float) -> np.ndarray:
    """Por evento: hay una referencia r con evento - r en [lo, hi] s."""
    if not len(events) or not len(refs):
        return np.zeros(len(events), dtype=bool)
    refs = np.sort(refs)
    lo, hi = events - int(hi_s * FPS), events - int(lo_s * FPS)
    return np.searchsorted(refs, hi, side="right") > np.searchsorted(refs, lo, side="left")


def gate(starts, role, stops, restarts, short, live_sure, graded, rng_seed=0, n_random=50) -> dict:
    """Agregados de la puerta de L2."""
    minutes = graded.sum() / FPS / 60

    def precision(ev):
        hit = _near(ev, stops, -1.0, 2.0) | _near(ev, restarts, -1.5, 1.5)
        return float(hit.mean()) if len(ev) else float("nan")

    rng = np.random.default_rng(rng_seed)
    pool = np.flatnonzero(graded)
    rand = [precision(np.sort(rng.choice(pool, size=len(starts), replace=False))) for _ in range(n_random)] \
        if len(starts) else [float("nan")]
    prec = precision(starts)
    cov_stop = _near(stops, starts, -2.0, 2.0).mean() if len(stops) else float("nan")  # silbato en +-2 s del inicio
    cov_rest = _near(restarts, starts, -2.0, 2.0).mean() if len(restarts) else float("nan")
    cov_any = (_near(stops, starts, -2.0, 2.0) | _near(restarts, starts, -2.0, 2.0)).mean() if len(stops) else float("nan")
    in_live = live_sure[starts].mean() if len(starts) else float("nan")
    live_minutes = live_sure.sum() / FPS / 60
    stop_role = role == 1
    short_mid = np.array([(s + e) // 2 for s, e in short])
    short_hit = (_near(short_mid, starts, -3.0, 3.0).mean() if len(short_mid) else float("nan"))
    rep = dict(
        silbatos=int(len(starts)), por_minuto=float(len(starts) / minutes),
        precision=prec, precision_azar=float(np.nanmean(rand)),
        precision_sobre_azar=float(prec / np.nanmean(rand)) if np.nanmean(rand) > 0 else float("nan"),
        cobertura_D_seguros=float(cov_any), cobertura_inicio=float(cov_stop), cobertura_reanudacion=float(cov_rest),
        D_seguros=int(len(stops)),
        frac_en_vivo_seguro=float(in_live),
        falsos_en_vivo_por_min=float(live_sure[starts].sum() / live_minutes) if live_minutes else float("nan"),
        frac_parada=float(stop_role.mean()) if len(role) else float("nan"),
        frac_reanudacion=float((role == -1).mean()) if len(role) else float("nan"),
        paradas_en_inicio_D=float(_near(starts[stop_role], stops, -1.0, 2.0).mean()) if stop_role.any() else float("nan"),
        reanudaciones_en_final_D=float(_near(starts[role == -1], restarts, -1.5, 1.5).mean()) if (role == -1).any() else float("nan"),
        D_cortos=int(len(short)), D_cortos_con_silbato=float(short_hit),
    )
    rep["pasa"] = bool(rep["precision_sobre_azar"] >= 3 and prec >= 0.5 and cov_any >= 0.4)
    return rep


def main(argv=None) -> int:
    import argparse

    from hackbarca.audio.whistle import half_offsets, read_audio
    from hackbarca.eval.evaluate import pm
    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED, collective_speed

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=SCORE_MIN)
    args = parser.parse_args(argv)

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    audio = read_audio(os.path.join(data_dir, "broadcast", "BAR-ATH_broadcast.mp4"))
    offsets = half_offsets(np.load(os.path.join(data_dir, "derived", "broadcast_alignment.npz"))["tactical_to_broadcast"],
                           GRADED)
    report = {"lineas_fijas": stationary_lines(audio)}
    t, score, pitch = whistle_score(audio)
    del audio

    match = load_match()
    n = match.n_frames
    speed = collective_speed(match)
    pred = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", "La MasIA.csv")))
    pmax = np.load(os.path.join(data_dir, "derived", "proba_l3.npz"))["proba"].max(1)
    graded = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    live_sure = graded & (pred <= 1) & (pmax >= 0.9)
    stops, restarts, short = reference_events(pred, pmax, GRADED)

    for thr in sorted(set(THRESHOLDS) | {args.threshold}):
        whistles = find_whistles(t, score, thr)
        starts = to_tactical_frames(whistles, offsets)
        starts = starts[(starts >= 0) & (starts < n)]
        starts = starts[graded[starts]]
        role = roles(starts, speed)
        report["umbral_%g" % thr] = gate(starts, role, stops, restarts, short, live_sure, graded)
        if thr == args.threshold:
            hz = [float(np.median(pitch[(t >= a) & (t <= b)])) for a, b, _ in whistles]
            report["tono_mediano_hz"] = float(np.median(hz)) if hz else float("nan")
            np.savez_compressed(os.path.join(data_dir, "derived", "whistles_v2.npz"), start=starts, role=role)

    for key, value in report.items():
        print("\n" + key)
        items = value.items() if isinstance(value, dict) else [("valor", value)]
        for k, v in items:
            print("   {:<28} {}".format(k, round(v, 3) if isinstance(v, float) else v))
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("\nguardado", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
