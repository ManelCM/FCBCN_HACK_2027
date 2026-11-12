"""Familia "control40" (E099): control sostenido + respuesta colectiva, 40 columnas.

Reproduce, sobre la entrega vigente (E075), el candidato que otra linea del equipo midio contra E035
en el banco v3 (+0,017, 9 de 10). Su informe describe 40 columnas; aqui se reimplementan desde esa
descripcion, sin su codigo:

32 de control y proximidad sostenida (16 por equipo):
  ctl_dist        distancia del jugador mas cercano del equipo al balon;
  ctl_margin      distancia del rival mas cercano - la propia (ventaja sobre el rival);
  ctl_close       proximidad clara: el mas cercano a < CLOSE_DIST y con > CLOSE_MARGIN de ventaja;
  ctl_close_fraction1/3   fraccion de proximidad clara en ventanas centradas de 1 y 3 s;
  ctl_close_run   duracion (log) de la racha de proximidad clara a la que pertenece el frame;
  ctl_since_close tiempo (log) desde la ultima proximidad clara;
  ctl_relspeed    velocidad relativa entre el balon y el jugador mas cercano;
  ctl_sup{2,5}_{b,f}{1,3}  soporte del equipo alrededor del balon (suma gaussiana de radio 2 y 5 m)
                  en ventanas pasadas / futuras de 1 y 3 s.
8 de respuesta colectiva (4 por equipo):
  response_decel  fraccion de jugadores que desaceleran (< DECEL m/s^2);
  response_dspeed1/3  velocidad media del equipo despues - antes, en 1 y 3 s;
  response_coherence  coherencia direccional: |media de los vectores unitarios| de los que se mueven.

Donde no se ve el balon, las de control son NaN y las ventanas promedian solo lo observado. Las
ventanas no cruzan el descanso (`match.period`). Una velocidad de slot > MAX_SPEED es un salto por
reasignacion de track y cuenta como dato que falta. Todo va en pares _h/_a: `swap_teams` lo
intercambia sin listas aparte.
"""

from __future__ import annotations

import warnings

import numpy as np

from hackbarca.features.tactical import _span_mean
from hackbarca.io.load_match import AWAY, HOME, MatchTracking

CLOSE_DIST = 3.0     # m
CLOSE_MARGIN = 0.5   # m
SUPPORT_RADII = (2.0, 5.0)
DECEL = -1.0         # m/s^2
MOVING = 1.0         # m/s, para la coherencia
MAX_SPEED = 15.0     # m/s
VEL_HALF = 5         # frames de la derivada centrada


def _segments(match: MatchTracking):
    """Tramos [a, b) de un mismo periodo (todo el partido si no hay periodo)."""
    if match.period is None:
        return [(0, match.n_frames)]
    edges = np.flatnonzero(np.diff(match.period.astype(np.int16))) + 1
    bounds = np.concatenate(([0], edges, [match.n_frames]))
    return list(zip(bounds[:-1], bounds[1:]))


def _seg_mean(x: np.ndarray, lo: int, hi: int, segs) -> np.ndarray:
    """_span_mean sin cruzar de un periodo a otro."""
    out = np.full(len(x), np.nan)
    for a, b in segs:
        out[a:b] = _span_mean(x[a:b], lo, hi)
    return out


def _velocity(x: np.ndarray, fps: int, segs) -> np.ndarray:
    """Derivada centrada en +-VEL_HALF frames dentro de cada periodo (m/s), por columnas."""
    v = np.full(x.shape, np.nan)
    h = VEL_HALF
    for a, b in segs:
        if b - a > 2 * h:
            v[a + h:b - h] = (x[a + 2 * h:b] - x[a:b - 2 * h]) * fps / (2 * h)
    return v


def _run_length(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mask))
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = b - a
    return out


def _since_last(mask: np.ndarray) -> np.ndarray:
    idx = np.where(mask, np.arange(len(mask)), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, np.arange(len(mask)) - last, 10_000)


def control40_features(match: MatchTracking, bx: np.ndarray, by: np.ndarray) -> dict:
    fps = match.fps
    segs = _segments(match)
    w1, w3 = fps, 3 * fps
    known = np.isfinite(bx) & np.isfinite(by)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.hypot(match.players_x - bx[:, None], match.players_y - by[:, None])
        pvx, pvy = _velocity(match.players_x, fps, segs), _velocity(match.players_y, fps, segs)
        pspeed = np.hypot(pvx, pvy)
        jump = pspeed > MAX_SPEED
        pvx[jump], pvy[jump], pspeed[jump] = np.nan, np.nan, np.nan
        bvx, bvy = _velocity(bx[:, None], fps, segs)[:, 0], _velocity(by[:, None], fps, segs)[:, 0]
        nearest = {}
        for team in (HOME, AWAY):
            cols = np.flatnonzero(match.team_of_track == team)
            d = np.where(np.isnan(dist[:, cols]), np.inf, dist[:, cols])
            k = np.argmin(d, axis=1)
            dmin = d[np.arange(match.n_frames), k]
            nearest[team] = (np.where(known & np.isfinite(dmin), dmin, np.nan), cols[k], cols)
        for team, name in ((HOME, "h"), (AWAY, "a")):
            other = AWAY if team == HOME else HOME
            d_own, slot, cols = nearest[team]
            d_opp = nearest[other][0]
            margin = np.clip(d_opp - d_own, -20, 20)
            close = (d_own < CLOSE_DIST) & (margin > CLOSE_MARGIN)
            close_f = np.where(known, close.astype(float), np.nan)
            out["ctl_dist_" + name] = d_own
            out["ctl_margin_" + name] = margin
            out["ctl_close_" + name] = close_f
            out["ctl_close_fraction1_" + name] = _seg_mean(close_f, -(w1 // 2), w1 // 2, segs)
            out["ctl_close_fraction3_" + name] = _seg_mean(close_f, -(w3 // 2), w3 // 2, segs)
            out["ctl_close_run_" + name] = np.log1p(_run_length(close))
            out["ctl_since_close_" + name] = np.log1p(_since_last(close))
            rows = np.arange(match.n_frames)
            out["ctl_relspeed_" + name] = np.where(
                known, np.hypot(bvx - pvx[rows, slot], bvy - pvy[rows, slot]), np.nan)
            d_team = dist[:, cols]
            for r in SUPPORT_RADII:
                sup = np.nansum(np.exp(-d_team ** 2 / (2 * r * r)), axis=1)
                sup = np.where(known, sup, np.nan)
                for tag, (lo, hi) in (("b1", (-w1, -1)), ("b3", (-w3, -1)), ("f1", (1, w1)), ("f3", (1, w3))):
                    out["ctl_sup%d_%s_%s" % (r, tag, name)] = _seg_mean(sup, lo, hi, segs)

            team_speed = pspeed[:, cols]
            seen = np.isfinite(team_speed)
            n_seen = seen.sum(1)
            mean_speed = np.where(n_seen > 0, np.nanmean(team_speed, axis=1), np.nan)
            accel = np.full(team_speed.shape, np.nan)
            h = VEL_HALF
            for a, b in segs:
                if b - a > 2 * h:
                    accel[a + h:b - h] = (team_speed[a + 2 * h:b] - team_speed[a:b - 2 * h]) * fps / (2 * h)
            n_acc = np.isfinite(accel).sum(1)
            out["response_decel_" + name] = np.where(n_acc > 0, (accel < DECEL).sum(1) / np.maximum(n_acc, 1),
                                                     np.nan)
            for tag, w in (("1", w1), ("3", w3)):
                out["response_dspeed%s_%s" % (tag, name)] = (_seg_mean(mean_speed, 1, w, segs)
                                                             - _seg_mean(mean_speed, -w, -1, segs))
            moving = team_speed > MOVING
            ux = np.where(moving, pvx[:, cols] / team_speed, 0.0)
            uy = np.where(moving, pvy[:, cols] / team_speed, 0.0)
            n_mov = moving.sum(1)
            out["response_coherence_" + name] = np.where(
                n_mov >= 3, np.hypot(ux.sum(1), uy.sum(1)) / np.maximum(n_mov, 1), np.nan)
    return out
