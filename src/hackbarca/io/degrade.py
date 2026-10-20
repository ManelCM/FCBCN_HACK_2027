"""Simulador de degradacion tipo ATD: tracking limpio -> tracking "automatico" (ADR-0006).

Lo que el ATD tiene y el tracking corregido no (CHALLENGE.md / DATA.md): balon ausente en ~41 % de
los frames evaluados, tracks fragmentados en slots sin identidad, huecos y ruido de posicion.
Todo es aleatorio con semilla fija: mismo `seed` -> mismo partido degradado.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hackbarca.io.load_match import MatchTracking


@dataclass(frozen=True)
class DegradeParams:
    # Balon: proceso on/off (rachas) que borra el balon visible; mas probable cerca de jugadores.
    ball_drop_rate: float = 0.30       # fraccion objetivo de frames con balon visible que se borran
    ball_gap_mean: float = 20.0        # duracion media de una racha sin balon (frames)
    occlusion_radius: float = 2.0      # m: balon a menos de esto de un jugador = ocluible
    occlusion_boost: float = 3.0       # cuanto mas probable es empezar un hueco si esta ocluido
    ball_noise: float = 0.3            # m, desviacion tipica del ruido del balon
    # En el tracking limpio el balon falta casi solo con balon muerto; el automatico lo sigue
    # viendo en parte de los parones. Prob. de que cada racha sin balon pase a visible.
    dead_ball_visible: float = 0.0
    # Jugadores: tracklets con huecos, reasignados a slots sin identidad.
    tracklet_mean: float = 750.0       # frames (30 s)
    player_gap_mean: float = 25.0      # frames
    player_noise: float = 0.2          # m
    team_flip_prob: float = 0.0        # prob. de que un tracklet salga con el equipo cambiado
    slots_per_team: int = 23           # 46 slots en el ATD tactico


def _bursty_mask(rng, n: int, start_prob: np.ndarray, gap_mean: float) -> np.ndarray:
    """Mascara True en rachas: en cada frame libre empieza un hueco con prob. start_prob[t]."""
    mask = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(rng.random(n) < start_prob)
    lengths = rng.geometric(1.0 / max(gap_mean, 1.0), size=len(starts))
    end_prev = 0
    for s, length in zip(starts, lengths):
        if s < end_prev:
            continue
        mask[s:s + length] = True
        end_prev = s + length
    return mask


def _reveal_gaps(rng, ball_x, ball_y, missing, prob, noise):
    """Rellena, con prob. `prob` por racha, los huecos originales interpolando sus extremos."""
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0 or stop >= len(ball_x) or rng.random() >= prob:
            continue
        a, b = start - 1, stop
        if np.isnan(ball_x[a]) or np.isnan(ball_x[b]):
            continue
        frac = (np.arange(start, stop) - a) / float(b - a)
        ball_x[start:stop] = ball_x[a] + frac * (ball_x[b] - ball_x[a]) + rng.normal(0, noise, stop - start)
        ball_y[start:stop] = ball_y[a] + frac * (ball_y[b] - ball_y[a]) + rng.normal(0, noise, stop - start)


def _nearest_distance(match: MatchTracking) -> np.ndarray:
    dx = match.players_x - match.ball_x[:, None]
    dy = match.players_y - match.ball_y[:, None]
    with np.errstate(invalid="ignore"):
        return np.nanmin(np.sqrt(dx * dx + dy * dy), axis=1, initial=np.inf)


def degrade(match: MatchTracking, params: DegradeParams = DegradeParams(),
            seed: int = 0) -> MatchTracking:
    rng = np.random.default_rng(seed)
    n = match.n_frames

    # --- balon ---
    visible = ~np.isnan(match.ball_x)
    occluded = visible & (_nearest_distance(match) < params.occlusion_radius)
    weight = np.where(occluded, params.occlusion_boost, 1.0)
    # p tal que la fraccion borrada ~ ball_drop_rate: rate ~= p * gap / (1 + p * gap)
    rate = min(params.ball_drop_rate, 0.95)
    base = rate / ((1.0 - rate) * params.ball_gap_mean * np.mean(weight[visible]) + 1e-12)
    drop = _bursty_mask(rng, n, np.clip(base * weight, 0, 1), params.ball_gap_mean)
    ball_x = match.ball_x + rng.normal(0, params.ball_noise, n)
    ball_y = match.ball_y + rng.normal(0, params.ball_noise, n)
    ball_x[drop] = np.nan
    ball_y[drop] = np.nan
    if params.dead_ball_visible > 0:
        _reveal_gaps(rng, ball_x, ball_y, ~visible, params.dead_ball_visible, params.ball_noise)

    # --- jugadores: tracklets -> slots ---
    px = match.players_x + rng.normal(0, params.player_noise, match.players_x.shape)
    py = match.players_y + rng.normal(0, params.player_noise, match.players_y.shape)
    n_slots = 2 * params.slots_per_team
    out_x = np.full((n, n_slots), np.nan)
    out_y = np.full((n, n_slots), np.nan)
    slot_team = np.repeat(np.array([0, 1], dtype=np.int8), params.slots_per_team)
    free_at = np.zeros(n_slots, dtype=np.int64)  # primer frame en que el slot queda libre

    tracklets = []
    for player in range(match.n_slots):
        present = np.flatnonzero(~np.isnan(px[:, player]))
        if not len(present):
            continue
        t, last = int(present[0]), int(present[-1])
        while t <= last:
            length = int(rng.geometric(1.0 / params.tracklet_mean))
            stop = min(t + length, last + 1)
            team = int(match.team_of_track[player])
            if rng.random() < params.team_flip_prob:
                team = 1 - team
            tracklets.append((t, stop, player, team))
            t = stop + int(rng.geometric(1.0 / max(params.player_gap_mean, 1.0)))

    for start, stop, player, team in sorted(tracklets):
        candidates = np.flatnonzero((slot_team == team) & (free_at <= start))
        if not len(candidates):
            continue  # sin slot libre: el tracklet se pierde, como en un tracker saturado
        slot = candidates[rng.integers(len(candidates))]
        out_x[start:stop, slot] = px[start:stop, player]
        out_y[start:stop, slot] = py[start:stop, player]
        free_at[slot] = stop

    return replace(match, team_of_track=slot_team, players_x=out_x, players_y=out_y,
                   players_speed=np.full_like(out_x, np.nan), ball_x=ball_x, ball_y=ball_y,
                   ball_speed=np.full(n, np.nan))
