"""Enlace temporal de detecciones de balon: de candidatos ruidosos por frame a una trayectoria.

Viterbi sobre (candidatos del frame + estado "sin balon"): premia la confianza del detector y
penaliza saltos fisicamente imposibles. Puro numpy, sin dependencias de vision.
"""

from __future__ import annotations

import numpy as np

NONE_COST = 1.5     # coste por frame de declarar "no hay balon"
BIRTH_COST = 1.5    # coste de pasar de "sin balon" a un candidato
BRIDGE = 6          # frames: huecos cortos entre detecciones enlazadas que se interpolan
MAX_SPEED = 40.0    # m/s: por encima, la transicion es imposible
SPEED_SCALE = 12.0  # m/s: escala del coste cuadratico de velocidad


def link_candidates(cands: list[np.ndarray], fps: int = 25) -> np.ndarray:
    """cands[t] = array (k_t, 3) con x, y (m) y confianza. Devuelve (n, 2) con NaN sin balon.

    Se asume que los frames son consecutivos; para tramos separados, llamar una vez por tramo.
    """
    n = len(cands)
    out = np.full((n, 2), np.nan)
    if n == 0:
        return out
    # estado 0 = sin balon; estados 1..k = candidatos
    costs = [np.concatenate(([NONE_COST], -np.log(np.clip(c[:, 2], 1e-3, 1.0)))) for c in cands]
    acc = costs[0].copy()
    acc[1:] += BIRTH_COST
    back = []
    for t in range(1, n):
        prev, cur = cands[t - 1], cands[t]
        trans = np.zeros((len(prev) + 1, len(cur) + 1))
        trans[0, 1:] = BIRTH_COST
        if len(prev) and len(cur):
            d = np.hypot(prev[:, None, 0] - cur[None, :, 0], prev[:, None, 1] - cur[None, :, 1])
            speed = d * fps
            trans[1:, 1:] = np.where(speed <= MAX_SPEED, 0.5 * (speed / SPEED_SCALE) ** 2, np.inf)
        total = acc[:, None] + trans
        b = np.argmin(total, axis=0)
        back.append(b)
        acc = total[b, np.arange(total.shape[1])] + costs[t]
    state = int(np.argmin(acc))
    for t in range(n - 1, -1, -1):
        if state > 0:
            out[t] = cands[t][state - 1, :2]
        if t > 0:
            state = int(back[t - 1][state])
    return _bridge(out, fps)


def _bridge(path: np.ndarray, fps: int) -> np.ndarray:
    """Interpola huecos de hasta BRIDGE frames si la velocidad implicita es fisica."""
    path = path.copy()
    have = np.flatnonzero(~np.isnan(path[:, 0]))
    for a, b in zip(have[:-1], have[1:]):
        gap = b - a
        if 1 < gap <= BRIDGE + 1 and np.hypot(*(path[b] - path[a])) * fps / gap <= MAX_SPEED:
            frac = (np.arange(a + 1, b) - a)[:, None] / gap
            path[a + 1:b] = path[a] + frac * (path[b] - path[a])
    return path
