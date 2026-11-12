"""A3 (E076): ¿se oye el golpeo del balon en el audio del broadcast? Solo agregados (ADR-0004).

    scripts/run_aggregate.sh kick uv run --group audio python -m hackbarca.audio.kick_gate

Salford/FascinatE: el golpeo es un impulso corto y de banda ancha; se ve mejor en la tasa de cambio de
la energia por banda (dE/dt) que en el espectrograma. Aqui: STFT a 100 tramas/s, flujo espectral
positivo en 100-4000 Hz (onset strength) y su maximo por frame de video, llevado al timeline tactico
con el desfase por parte (E026). Puerta: AUC del maximo en +-2 frames alrededor de
  - toques del ATD (pico de dv del balon >= 9 m/s, el TOUCH_DV de frame_features), y
  - cambios H<->A directos de nuestra prediccion segura (pmax >= 0,9 a ambos lados),
frente a instantes al azar de juego vivo predicho. Q6: solo validacion.
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.audio.whistle import read_audio
from hackbarca.eval.evaluate import pm
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.real_signals import _data, _offsets, auc

RATE = 16000
HOP = 160          # 10 ms
NFFT = 512
BAND = (100.0, 4000.0)
FPS = 25


def onset_strength(audio: np.ndarray) -> np.ndarray:
    """Flujo espectral positivo (log-magnitud) en BAND por trama de 10 ms."""
    window = np.hanning(NFFT).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / RATE)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    n = (len(audio) - NFFT) // HOP
    out = np.zeros(n, dtype=np.float32)
    prev = None
    for a in range(0, n, 20000):
        idx = np.arange(a, min(a + 20000, n))[:, None] * HOP + np.arange(NFFT)[None, :]
        mag = np.log1p(np.abs(np.fft.rfft(audio[idx] * window, axis=1))[:, band])
        diff = np.diff(mag, axis=0, prepend=mag[:1] if prev is None else prev[None, :])
        out[a:a + len(mag)] = np.maximum(diff, 0).sum(1)
        prev = mag[-1]
    return out


def main() -> int:
    from hackbarca.features.frame_features import TOUCH_DV, TOUCH_W
    from hackbarca.io.load_match import load_match
    strength = onset_strength(read_audio(os.path.join(_data("broadcast"), "BAR-ATH_broadcast.mp4"), RATE))
    # maximo por frame de video (4 tramas de 10 ms por frame de 40 ms) en el reloj del broadcast
    nb = len(strength) // 4
    per_frame = strength[:nb * 4].reshape(nb, 4).max(1)
    match = load_match()
    n = match.n_frames
    series = np.full(n, np.nan)
    for lo, hi, offset in _offsets():
        a, b = max(lo, 0), min(hi, nb)
        series[a + offset:b + offset] = per_frame[a:b]
    # normalizar por minuto (el nivel del audio cambia con la grada y el comentario)
    med = np.full(n, np.nan)
    for s in range(0, n, 1500):
        chunk = series[s:s + 1500]
        if np.isfinite(chunk).any():
            med[s:s + 1500] = np.nanmedian(chunk)
    rel = series / med
    peak = np.full(n, np.nan)
    peak[2:-2] = np.nanmax(np.stack([rel[k:n - 4 + k] for k in range(5)]), axis=0)

    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    pmax = np.load(_data("derived", "proba_l3.npz"))["proba"].max(1)
    graded = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    live = graded & ((pred == 0) | (pred == 1))
    rng = np.random.default_rng(0)
    rand = rng.choice(np.flatnonzero(live), 5000, replace=False)
    # toques del ATD (balon visible): pico de dv como en frame_features (version con umbral fijo)
    bx, by = match.ball_x, match.ball_y
    v = np.full((n, 2), np.nan)
    v[TOUCH_W:-TOUCH_W, 0] = (bx[2 * TOUCH_W:] - bx[:-2 * TOUCH_W]) * FPS / (2 * TOUCH_W)
    v[TOUCH_W:-TOUCH_W, 1] = (by[2 * TOUCH_W:] - by[:-2 * TOUCH_W]) * FPS / (2 * TOUCH_W)
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[TOUCH_W:] = v[:-TOUCH_W]
    after[:-TOUCH_W] = v[TOUCH_W:]
    dv = np.hypot(after[:, 0] - before[:, 0], after[:, 1] - before[:, 1])
    touches = np.flatnonzero(live & np.isfinite(dv) & (dv > TOUCH_DV))
    changes = []
    for a, b in GRADED:
        w = pred[a:b]
        for t in np.flatnonzero(w[1:] != w[:-1]) + 1:
            f = a + t
            if {int(w[t - 1]), int(w[t])} == {0, 1} and pmax[f - 25:f].min() >= 0.9 and pmax[f:f + 25].min() >= 0.9:
                changes.append(f)
    changes = np.array(changes, dtype=int)
    base = peak[rand]
    print("cobertura del audio en frames puntuados: %.3f" % np.isfinite(series[graded]).mean())
    print("pico relativo (mediana) | azar vivo %.2f (n=%d)" % (np.nanmedian(base), len(rand)))
    for label, frames in (("toques del ATD (dv > %g m/s)" % TOUCH_DV, touches), ("cambios H<->A seguros", changes)):
        vals = peak[frames] if len(frames) else np.zeros(0)
        print("%-32s n=%5d  mediana %.2f  AUC frente al azar %.3f" % (label, len(frames), np.nanmedian(vals) if len(vals) else np.nan,
                                                                     auc(vals, base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
