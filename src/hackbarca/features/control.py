"""Familia "control" (M2a, E051-E052): quien controla el balon, no solo quien esta mas cerca.

CHALLENGE.md define la posesion por el **control** del balon, y lo que mas fallamos son posesiones
cortas sin pases y robos con el balon visible (E040: +0,097). "El jugador mas cercano" acierta solo el
0,715 con el balon degradado: estar cerca un instante no es controlar. Aqui, por equipo:

  control      = balon visible, a < CTRL_DIST de un jugador del equipo, mas cerca que el rival mas
                 cercano y a menos de CTRL_SPEED (no un balon que pasa de largo);
  ctrl_run     = duracion (log) de la racha de control en curso;
  ctrl_b/f     = fraccion de frames con control en los 1 y 3 s anteriores / posteriores y su cambio
                 (un robo se ve en el cambio, como en las familias de tactical.py);
  ctrl_since/until = tiempo (log) desde el ultimo control y hasta el siguiente;
  ctrl_switch  = cambios de jugador controlador del mismo equipo en 3 s (aproximacion a los pases;
                 los slots no tienen identidad, pero dentro de un tracklet si son estables);
  ctrl_press   = rivales a < PRESS_DIST del balon mientras el equipo lo controla.
Todo va en pares _h/_a, asi que `swap_teams` lo intercambia sin listas aparte.
"""

from __future__ import annotations

import warnings

import numpy as np

from hackbarca.io.load_match import AWAY, HOME, MatchTracking

CTRL_DIST = 1.5    # m
CTRL_SPEED = 4.0   # m/s
PRESS_DIST = 2.0   # m
WINDOWS = (25, 75)


def _window_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Media de x en [t+lo, t+hi] recortada a los bordes."""
    n = len(x)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    a, b = np.clip(idx + lo, 0, n), np.clip(idx + hi + 1, 0, n)
    return (cs[b] - cs[a]) / np.maximum(b - a, 1)


def _since_last(mask: np.ndarray) -> np.ndarray:
    """Frames desde el ultimo True (incluido el actual = 0); 1e4 si no hubo ninguno."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, np.arange(len(mask)) - last, 10_000)


def _run_length(mask: np.ndarray) -> np.ndarray:
    """Longitud total de la racha True a la que pertenece cada frame (0 fuera)."""
    out = np.zeros(len(mask))
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = b - a
    return out


def control_masks(match: MatchTracking, bx: np.ndarray, by: np.ndarray, known: np.ndarray):
    """Por equipo: (control, slot controlador o -1, rivales cerca). bx/by con NaN donde no se ve."""
    n, fps = match.n_frames, match.fps
    speed = np.full(n, np.nan)
    speed[2:-2] = np.hypot(bx[4:] - bx[:-4], by[4:] - by[:-4]) * fps / 4
    slow = known & (np.nan_to_num(speed, nan=np.inf) < CTRL_SPEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.hypot(match.players_x - bx[:, None], match.players_y - by[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    best = {}
    for team in (HOME, AWAY):
        cols = np.flatnonzero(match.team_of_track == team)
        d = dist[:, cols]
        k = np.argmin(d, axis=1) if len(cols) else np.zeros(n, dtype=int)
        best[team] = (d[np.arange(n), k] if len(cols) else np.full(n, np.inf), cols[k] if len(cols) else k)
    out = {}
    for team in (HOME, AWAY):
        other = AWAY if team == HOME else HOME
        d_own, slot = best[team]
        ctrl = slow & (d_own < CTRL_DIST) & (d_own < best[other][0])
        opp = dist[:, match.team_of_track == other]
        press = (opp < PRESS_DIST).sum(1).astype(float)
        out[team] = (ctrl, np.where(ctrl, slot, -1), press)
    return out


def control_features(match: MatchTracking, bx: np.ndarray, by: np.ndarray, known: np.ndarray) -> dict:
    f = {}
    masks = control_masks(match, np.where(known, bx, np.nan), np.where(known, by, np.nan), known)
    for team, name in ((HOME, "h"), (AWAY, "a")):
        ctrl, slot, press = masks[team]
        c = ctrl.astype(float)
        f["ctrl_" + name] = c
        f["ctrl_run_" + name] = np.log1p(_run_length(ctrl))
        for w in WINDOWS:
            before = _window_mean(c, -w, -1)
            after = _window_mean(c, 1, w)
            f["ctrl_b%d_%s" % (w, name)] = before
            f["ctrl_f%d_%s" % (w, name)] = after
            f["ctrl_chg%d_%s" % (w, name)] = after - before
        f["ctrl_since_" + name] = np.log1p(_since_last(ctrl))
        f["ctrl_until_" + name] = np.log1p(_since_last(ctrl[::-1])[::-1])
        # cambio de controlador dentro del equipo: slot distinto al del control anterior del equipo
        prev = np.full(len(slot), -1)
        idx = np.flatnonzero(slot >= 0)
        if len(idx) > 1:
            prev[idx[1:]] = slot[idx[:-1]]
        switch = ((slot >= 0) & (prev >= 0) & (slot != prev)).astype(float)
        f["ctrl_switch_" + name] = _window_mean(switch, -75, 75) * 151
        f["ctrl_press_" + name] = np.where(ctrl, press, np.nan)
    return f
