"""Puerta C2: silbato, grada y comentario con un modelo preentrenado en AudioSet (PANNs), no con umbrales.

E029 no encontro silbato con procesado de senal (tonalidad en 2-4,5 kHz). Aqui se prueba lo que usa
la literatura: PANNs (Kong et al., 2020), CNN preentrenada en AudioSet (527 clases) con salida por
frame (~100 por segundo), que trae las clases "Whistle", "Cheering", "Crowd", "Applause" y "Speech".

Solo valida (Q6 sin responder: el audio no entra en la entrega). Sin GT del reto, se compara con
nuestra prediccion de la entrega:
  - silbato alrededor del inicio de los D largos predichos (>= 5 s; casi seguro un paron real),
    de los D cortos predichos (< 5 s; sospechosos, E041) y de las reanudaciones, frente a instantes
    al azar de juego vivo: AUC por rangos;
  - grada (Cheering + Crowd + Applause) media en juego vivo frente a D.
La salida por frame se queda en data/derived/audioset.npz; el agente no la lee. Solo agregados.

    scripts/run_aggregate.sh audioset uv run --group audio --group vision --with panns-inference \
        python -m hackbarca.audio.audioset_gate
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.audio.whistle import half_offsets, read_audio

RATE = 32000
CHUNK_S = 60
CLASSES = ("Whistle", "Cheering", "Crowd", "Applause", "Speech")
H, A, D = 0, 1, 2
WIN = 12   # frames alrededor del instante (+-0,5 s)


def framewise(audio: np.ndarray, device: str = "cuda") -> tuple[np.ndarray, list[str]]:
    """Probabilidades por frame de video (25 por segundo) de CLASSES, por trozos de 60 s."""
    from panns_inference import SoundEventDetection, labels

    sed = SoundEventDetection(checkpoint_path=None, device=device)
    idx = [labels.index(c) for c in CLASSES]
    out = []
    step = CHUNK_S * RATE
    for a in range(0, len(audio), step):
        chunk = audio[a:a + step]
        if len(chunk) < RATE:
            break
        prob = sed.inference(chunk[None, :])[0][:, idx]          # (~100/s, k)
        n = int(round(len(chunk) / RATE * 25))
        edges = np.linspace(0, len(prob), n + 1).astype(int)
        out.append(np.array([prob[edges[i]:max(edges[i + 1], edges[i] + 1)].mean(0) for i in range(n)]))
    return np.concatenate(out), list(CLASSES)


def to_tactical_series(series: np.ndarray, offsets, n_frames: int) -> np.ndarray:
    """Serie por frame del broadcast -> por frame tactico con el desfase de cada parte."""
    out = np.full((n_frames, series.shape[1]), np.nan, dtype=np.float32)
    for lo, hi, offset in offsets:   # [lo, hi) en frames del broadcast; tactico = broadcast + offset
        a, b = max(lo, 0), min(hi, len(series))
        if b > a:
            out[a + offset:b + offset] = series[a:b]
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    ranks[np.argsort(allv, kind="mergesort")] = np.arange(1, len(allv) + 1)
    n1, n0 = float(len(pos)), float(len(neg))
    return float((ranks[:len(pos)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def onsets(codes: np.ndarray, runs, kind: str) -> np.ndarray:
    """Frames de inicio de D largo, D corto o reanudacion (D -> vivo) en la prediccion."""
    from hackbarca.eval.evaluate import pm
    out = []
    for a, b in runs:
        segs = pm.segments(codes, [(a, b)])
        for n, (s, e, c) in enumerate(segs):
            if c != D or n == 0:
                continue
            if kind == "long" and e - s >= 125:
                out.append(s)
            elif kind == "short" and e - s < 125 and n < len(segs) - 1:
                out.append(s)
            elif kind == "restart" and e - s >= 125 and n < len(segs) - 1:
                out.append(e)
    return np.array(out, dtype=int)


def peak_around(series: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.array([np.nanmax(series[max(f - WIN, 0):f + WIN + 1]) if np.isfinite(series[max(f - WIN, 0):f + WIN + 1]).any()
                     else np.nan for f in frames])


def main() -> int:
    from hackbarca.eval.evaluate import pm
    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    cache = os.path.join(data_dir, "derived", "audioset.npz")
    if not os.path.exists(cache):
        audio = read_audio(os.path.join(data_dir, "broadcast", "BAR-ATH_broadcast.mp4"), RATE)
        series, names = framewise(audio)
        np.savez_compressed(cache, series=series, names=np.array(names))
    blob = np.load(cache)
    series, names = blob["series"], [str(n) for n in blob["names"]]
    alignment = np.load(os.path.join(data_dir, "derived", "broadcast_alignment.npz"))
    offsets = half_offsets(alignment["tactical_to_broadcast"], GRADED)
    n_frames = load_match().n_frames
    tac = to_tactical_series(series, offsets, n_frames)
    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    runs = [tuple(r) for r in GRADED]
    graded = np.zeros(n_frames, dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    print("cobertura del audio sobre los frames puntuados: %.3f" % np.isfinite(tac[graded, 0]).mean())

    rng = np.random.default_rng(0)
    live = np.flatnonzero(graded & ((pred == H) | (pred == A)))
    rand = rng.choice(live, 3000, replace=False)
    wh = tac[:, names.index("Whistle")]
    print("\nSilbato (prob. maxima en +-0,5 s) frente a instantes al azar de juego vivo predicho")
    base = peak_around(wh, rand)
    print("   azar (vivo)                 n=%4d  mediana %.4f" % (len(rand), np.nanmedian(base)))
    for kind, label in (("long", "inicio de D largo (>= 5 s)"), ("short", "inicio de D corto (< 5 s)"),
                        ("restart", "reanudacion tras D largo")):
        f = onsets(pred, runs, kind)
        v = peak_around(wh, f)
        print("   {:<28} n={:4d}  mediana {:.4f}  AUC frente al azar {:.3f}".format(label, len(f), np.nanmedian(v), auc(v, base)))
    crowd = np.nansum(tac[:, [names.index(c) for c in ("Cheering", "Crowd", "Applause")]], axis=1)
    print("\nGrada media: juego vivo predicho %.4f | D predicho %.4f" % (
        np.nanmean(crowd[graded & (pred != D)]), np.nanmean(crowd[graded & (pred == D)])))
    sp = tac[:, names.index("Speech")]
    print("Habla media: juego vivo predicho %.4f | D predicho %.4f" % (
        np.nanmean(sp[graded & (pred != D)]), np.nanmean(sp[graded & (pred == D)])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
