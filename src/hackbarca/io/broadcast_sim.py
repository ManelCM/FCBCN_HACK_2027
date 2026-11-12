"""Broadcast sintetico a partir de un tracking tactico, con la correspondencia verdadera.

Imita lo que DATA.md describe del broadcast: planos con cortes, repeticiones (aqui a camara
lenta), primeros planos casi sin jugadores, vista parcial alrededor del balon y slots que no
siguen a nadie entre planos. Sirve para probar `broadcast_align` y para simular el relleno de
balon en el banco.
"""

from __future__ import annotations

import numpy as np

from hackbarca.io.load_match import MatchTracking


def simulate_broadcast(t: MatchTracking, seed: int = 0, view_radius: float = 30.0,
                       noise: float = 0.3, ball_visible: float = 0.8, n_slots: int = 51, true_ball=None):
    """(broadcast MatchTracking, truth) con truth[b] = frame tactico o -1 (repeticion/primer plano).

    `true_ball` = (x, y) del balon real: el broadcast es otro tracker y puede ver el balon donde
    el tactico lo pierde. Por defecto se usa el balon tactico.
    """
    true_x, true_y = true_ball if true_ball is not None else (t.ball_x, t.ball_y)
    rng = np.random.default_rng(seed)
    frames, truth_list, kinds = [], [], []
    pos = int(rng.integers(0, 250))
    while pos < t.n_frames - 50:
        length = int(rng.integers(75, 375))
        kind = rng.choice(["live", "live", "live", "live", "replay", "closeup"])
        if kind == "live":
            seg = np.arange(pos, min(pos + length, t.n_frames))
            frames.append(seg)
            truth_list.append(seg)
            pos = seg[-1] + 1 + int(rng.integers(0, 50))   # el corte puede saltarse frames
        elif kind == "replay" and pos > 500:
            start = pos - int(rng.integers(200, 500))
            seg = np.repeat(np.arange(start, start + length // 2), 2)  # camara lenta x2
            frames.append(seg)
            truth_list.append(np.full(len(seg), -1))
        else:
            seg = np.arange(pos, min(pos + 50, t.n_frames))
            frames.append(-seg - 1)                                       # marca primer plano
            truth_list.append(np.full(len(seg), -1))
        kinds.append(kind)
    src = np.concatenate(frames)
    truth = np.concatenate(truth_list)
    n = len(src)

    px = np.full((n, n_slots), np.nan)
    py = np.full((n, n_slots), np.nan)
    team = np.repeat(np.array([0, 1], dtype=np.int8), [n_slots // 2 + 1, n_slots // 2])
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    b = 0
    for seg_frames in frames:
        closeup = seg_frames[0] < 0
        tf = -seg_frames - 1 if closeup else seg_frames
        # slots nuevos en cada plano: permutacion de los slots de cada equipo
        perm = {0: rng.permutation(np.flatnonzero(team == 0)), 1: rng.permutation(np.flatnonzero(team == 1))}
        used = {0: 0, 1: 0}
        slot_of = {}
        for p in range(t.n_slots):
            k = int(t.team_of_track[p])
            if used[k] < len(perm[k]):
                slot_of[p] = perm[k][used[k]]
                used[k] += 1
        cx, cy = np.nanmean(true_x[tf]), np.nanmean(true_y[tf])
        if np.isnan(cx):
            cx, cy = t.pitch_length / 2, t.pitch_width / 2
        for i, f in enumerate(tf):
            row = b + i
            for p, s in slot_of.items():
                x, y = t.players_x[f, p], t.players_y[f, p]
                if np.isnan(x) or np.hypot(x - cx, y - cy) > (5.0 if closeup else view_radius):
                    continue
                px[row, s] = x + rng.normal(0, noise)
                py[row, s] = y + rng.normal(0, noise)
            if not closeup and rng.random() < ball_visible and not np.isnan(true_x[f]):
                bx[row] = true_x[f] + rng.normal(0, noise)
                by[row] = true_y[f] + rng.normal(0, noise)
        b += len(tf)

    broadcast = MatchTracking(n_frames=n, fps=t.fps, pitch_length=t.pitch_length,
                              pitch_width=t.pitch_width, team_of_track=team,
                              players_x=px, players_y=py, players_speed=np.full_like(px, np.nan),
                              ball_x=bx, ball_y=by, ball_speed=np.full(n, np.nan))
    return broadcast, truth
