"""Decodificacion offline H/A/D: Viterbi con coste por cambio de estado.

Al decodificar el partido entero de una vez, cada frontera se coloca donde mejor explica las
emisiones a ambos lados, no cuando un filtro causal "se convence": no hay retraso sistematico.
"""

from __future__ import annotations

import numpy as np

H, A, D = 0, 1, 2


def transition_costs(switch: float, to_dead: float, from_dead: float) -> np.ndarray:
    """Matriz 3x3 de costes (log-prob negativas) entre estados H, A, D."""
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = switch
    cost[H, D] = cost[A, D] = to_dead
    cost[D, H] = cost[D, A] = from_dead
    return cost


def viterbi(emission_cost: np.ndarray, trans_cost: np.ndarray) -> np.ndarray:
    """Camino de coste minimo. emission_cost: (n, 3) con -log p(obs | estado)."""
    n = emission_cost.shape[0]
    back = np.empty((n, 3), dtype=np.int8)
    acc = emission_cost[0].copy()
    for t in range(1, n):
        total = acc[:, None] + trans_cost          # (prev, cur)
        back[t] = np.argmin(total, axis=0)
        acc = total[back[t], np.arange(3)] + emission_cost[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmin(acc))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_runs(emission_cost: np.ndarray, trans_cost: np.ndarray, runs) -> np.ndarray:
    """Viterbi independiente en cada tramo [start, stop) (mitades); fuera se deja D."""
    out = np.full(emission_cost.shape[0], D, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = viterbi(emission_cost[start:stop], trans_cost)
    return out
