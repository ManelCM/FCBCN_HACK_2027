"""E123 (I7 de otros equipos, AndaAnda): un balón muerto que se lleva a sacar, no que deriva.

En el banco limpio el balón muerto casi siempre falta (en DFL, en todos los frames D). El simulador
(`degrade._reveal_gaps`) rellena una parte de esos huecos interpolando en línea recta entre el último punto
vivo y el primero de la reanudación: el balón deriva despacio y a velocidad constante durante todo el parón.
Eso se parece al «balón lento» del juego vivo (E043), el origen de los D cortos del reto.

En el ATD real, en un parón el balón se para o sale, un jugador lo recoge y lo lleva andando y se queda
quieto en el punto de saque hasta que se reanuda. Aquí, en los huecos revelados:
  1. quieto en el último punto (0,5-2 s, con el ruido del tracker);
  2. llevado al punto de reanudación a paso de jugador (1,2-2,5 m/s), con algo de vaivén lateral;
  3. quieto en el punto de saque hasta el final del hueco.
Si el hueco no da para llevarlo andando, se deja la interpolación de siempre.

Se usa solo para **copias extra de entrenamiento** (como AndaAnda): el banco que evalúa no cambia.
"""

from __future__ import annotations

import numpy as np

FPS = 25


def carry_reveal(rng, ball_x, ball_y, missing, prob, noise):
    """Igual que `degrade._reveal_gaps` (mismas rachas y misma probabilidad), con el balón llevado a sacar."""
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0 or stop >= len(ball_x) or rng.random() >= prob:
            continue
        a, b = start - 1, stop
        if np.isnan(ball_x[a]) or np.isnan(ball_x[b]):
            continue
        n = stop - start
        ax, ay, bx, by = ball_x[a], ball_y[a], ball_x[b], ball_y[b]
        dist = float(np.hypot(bx - ax, by - ay))
        wait = int(rng.uniform(0.5, 2.0) * FPS)
        walk = int(np.ceil(dist / rng.uniform(1.2, 2.5) * FPS))
        if wait + walk + FPS > n:          # no da para llevarlo andando: la interpolación de siempre
            frac = (np.arange(start, stop) - a) / float(b - a)
            xs, ys = ax + frac * (bx - ax), ay + frac * (by - ay)
        else:
            xs, ys = np.full(n, bx, dtype=float), np.full(n, by, dtype=float)
            xs[:wait], ys[:wait] = ax, ay
            f = (np.arange(walk) + 1) / walk
            sway = np.sin(np.linspace(0, np.pi * rng.uniform(2, 6), walk)) * rng.uniform(0.1, 0.4)
            ux, uy = (by - ay) / max(dist, 1e-6), -(bx - ax) / max(dist, 1e-6)   # perpendicular
            xs[wait:wait + walk] = ax + f * (bx - ax) + sway * ux
            ys[wait:wait + walk] = ay + f * (by - ay) + sway * uy
        ball_x[start:stop] = xs + rng.normal(0, noise, n)
        ball_y[start:stop] = ys + rng.normal(0, noise, n)


def degrade_with_carry(match, params, seed: int):
    """`degrade` con los huecos revelados al estilo de `carry_reveal` (el resto del simulador, igual)."""
    from hackbarca.io import degrade as module
    original = module._reveal_gaps
    module._reveal_gaps = carry_reveal
    try:
        return module.degrade(match, params, seed=seed)
    finally:
        module._reveal_gaps = original
