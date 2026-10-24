"""Errores de un CSV de entrega en la ventana publica de 5 min, frame a frame (autorizado, 24 oct).

La GT de la ventana publica (video frames 11185-18684) es publica y las predicciones son la salida de
nuestro algoritmo. Manel autorizo que el agente las vea frame a frame **solo dentro de esta ventana**
para analizar errores. Este script recorta a la ventana y no imprime nada de fuera.

    uv run python -m hackbarca.pipeline.public_window "La MasIA" [otro_csv ...]
"""

from __future__ import annotations

import os
import sys

import numpy as np

from hackbarca.eval.evaluate import pm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GT_PATH = os.path.join(_REPO_ROOT, "external", "more-than-a-hack-2027", "evaluation", "gt",
                       "BAR-ATH_possession_gt_public.csv")
WINDOW = (11185, 18685)   # [inicio, fin) en video frames: la ventana publica
LETTER = np.array(["H", "A", "D", "X"])


def clock(frame: int) -> str:
    """Tiempo de juego desde el inicio de la ventana (m:ss.d)."""
    s = (frame - WINDOW[0]) / 25
    return "%d:%04.1f" % (s // 60, s % 60)


def runs(codes: np.ndarray, offset: int):
    out, start = [], 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            out.append((offset + start, offset + i, int(codes[start])))
            start = i
    return out


def report(name: str) -> None:
    lo, hi = WINDOW
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))[lo:hi]
    pred = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", name + ".csv")))[lo:hi]
    print("=" * 100)
    print(name, "| acierto por frame %.3f" % np.mean(gt == pred))
    print("\nSegmentos reales (GT) frente a lo que predecimos dentro de cada uno:")
    for a, b, c in runs(gt, lo):
        seg = pred[a - lo:b - lo]
        mix = ", ".join("%s %d%%" % (LETTER[k], round(100 * np.mean(seg == k))) for k in (0, 1, 2) if np.any(seg == k))
        mark = "OK " if np.mean(seg == c) >= 0.8 else "MAL"
        print("  %s %s-%s  %-1s  %5.1f s   predicho: %s" % (mark, clock(a), clock(b), LETTER[c], (b - a) / 25, mix))
    print("\nNuestros segmentos:")
    print("  " + " | ".join("%s %s-%s" % (LETTER[c], clock(a), clock(b)) for a, b, c in runs(pred, lo)))
    print("\nCada cambio real y el cambio nuestro del mismo tipo mas cercano:")
    gt_t = [(a, int(gt[a - lo - 1]), c) for a, b, c in runs(gt, lo)[1:]]
    pr_t = [(a, int(pred[a - lo - 1]), c) for a, b, c in runs(pred, lo)[1:]]
    used = set()
    for f, x, y in gt_t:
        same = [(abs(g - f), g) for g, u, v in pr_t if v == y and (u == x or {x, y} == {0, 1})]
        best = min(same) if same else None
        if best:
            used.add(best[1])
        lag = "%+.1f s" % ((best[1] - f) / 25) if best else "no lo vemos"
        print("  %s  %s->%s   nuestro: %s" % (clock(f), LETTER[x], LETTER[y], lag))
    extra = [(g, u, v) for g, u, v in pr_t if g not in used]
    print("\nCambios nuestros que no corresponden a ninguno real (sobrantes):")
    print("  " + (", ".join("%s %s->%s" % (clock(g), LETTER[u], LETTER[v]) for g, u, v in extra) or "ninguno"))


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["La MasIA"]:
        report(arg)
