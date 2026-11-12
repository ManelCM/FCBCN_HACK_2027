"""Modelo L1: emisiones por reglas + Viterbi (histeresis con fronteras retrofechadas).

- Equipo: compara la distancia al balon del jugador mas cercano de cada equipo.
- Balon muerto: balon fuera del campo, o hueco sin balon mas largo que `max_gap`.
- Huecos cortos sin balon se interpolan: el balon no desaparece, el tracker lo pierde.
- La posesion no cambia salvo que la evidencia del rival compense `switch` (histeresis); al
  ser offline, la frontera cae donde empieza esa evidencia, no donde se confirma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hackbarca.decode.viterbi import decode_runs, transition_costs
from hackbarca.io.load_match import AWAY, HOME, MatchTracking

EPS = 1e-6


@dataclass(frozen=True)
class L1Params:
    max_gap: int = 25           # frames sin balon que se interpolan
    dist_scale: float = 1.5     # m: escala del logit de equipo sobre (d_rival - d_propio)
    p_dead_visible: float = 0.05
    p_dead_gap: float = 0.7     # prob. de muerto en un hueco largo sin balon
    p_dead_out: float = 0.97    # balon fuera del campo
    out_margin: float = 1.0     # m fuera de la linea para considerarlo fuera
    switch: float = 8.0         # coste de un cambio H<->A
    to_dead: float = 6.0
    from_dead: float = 6.0
    # Smart Tagging: ventana de ~5 s alrededor de cada reanudacion
    tag_before: int = 100       # frames antes de la ventana con D reforzado
    p_tag_before: float = 0.0   # 0 desactiva el uso de las etiquetas
    p_tag_in: float = 0.0


def apply_tags(p_dead: np.ndarray, windows, p: L1Params) -> np.ndarray:
    """Refuerza el balon muerto antes y dentro de cada ventana de reanudacion."""
    if not windows or p.p_tag_before <= 0:
        return p_dead
    p_dead = p_dead.copy()
    n = len(p_dead)
    for start, stop in windows:
        lo, a, b = max(0, start - p.tag_before), max(0, start), min(n, stop + 1)
        p_dead[lo:a] = np.maximum(p_dead[lo:a], p.p_tag_before)
        p_dead[a:b] = np.maximum(p_dead[a:b], p.p_tag_in)
    return p_dead


def _gap_lengths(missing: np.ndarray) -> np.ndarray:
    """Longitud del hueco al que pertenece cada frame (0 si hay dato)."""
    out = np.zeros(len(missing), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = stop - start
    return out


def interpolate_ball(match: MatchTracking, max_gap: int):
    """Balon con huecos <= max_gap interpolados linealmente; devuelve (x, y, gap_len)."""
    missing = np.isnan(match.ball_x)
    gap = _gap_lengths(missing)
    idx = np.arange(match.n_frames)
    x, y = match.ball_x.copy(), match.ball_y.copy()
    if (~missing).sum() >= 2:
        fill = missing & (gap <= max_gap)
        x[fill] = np.interp(idx[fill], idx[~missing], match.ball_x[~missing])
        y[fill] = np.interp(idx[fill], idx[~missing], match.ball_y[~missing])
    return x, y, gap


def team_distances(match: MatchTracking, bx: np.ndarray, by: np.ndarray):
    """(d_home, d_away): distancia del jugador mas cercano de cada equipo al balon (inf si no hay)."""
    dx = match.players_x - bx[:, None]
    dy = match.players_y - by[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, match.team_of_track == HOME].min(axis=1, initial=np.inf)
    d_away = dist[:, match.team_of_track == AWAY].min(axis=1, initial=np.inf)
    return d_home, d_away


def emission_probs(match: MatchTracking, p: L1Params, windows=None) -> np.ndarray:
    """(n, 3) probabilidades H, A, D por frame."""
    bx, by, gap = interpolate_ball(match, p.max_gap)
    known = ~np.isnan(bx)
    d_home, d_away = team_distances(match, np.nan_to_num(bx), np.nan_to_num(by))
    diff = np.clip(np.where(np.isfinite(d_home) & np.isfinite(d_away), d_away - d_home, 0.0), -50, 50)
    p_home = 1.0 / (1.0 + np.exp(-diff / p.dist_scale))
    p_home = np.where(known, p_home, 0.5)

    out = (bx < -p.out_margin) | (bx > match.pitch_length + p.out_margin) | \
          (by < -p.out_margin) | (by > match.pitch_width + p.out_margin)
    p_dead = np.where(known, np.where(out, p.p_dead_out, p.p_dead_visible), p.p_dead_gap)
    p_dead = apply_tags(p_dead, windows, p)

    probs = np.empty((match.n_frames, 3))
    probs[:, 0] = (1 - p_dead) * p_home
    probs[:, 1] = (1 - p_dead) * (1 - p_home)
    probs[:, 2] = p_dead
    return probs


def predict_l1(match: MatchTracking, runs, params: L1Params = L1Params(),
               windows=None) -> np.ndarray:
    """Codigos H/A/D (0/1/2) por frame; `runs` son los tramos evaluados [start, stop) y
    `windows` las ventanas de reanudacion del Smart Tagging (opcional)."""
    cost = -np.log(emission_probs(match, params, windows) + EPS)
    trans = transition_costs(params.switch, params.to_dead, params.from_dead)
    return decode_runs(cost, trans, runs)
