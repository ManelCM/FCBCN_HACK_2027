"""Familias "de entrenador y analista" del plan completo (T2): F3 firmas del robo, F5 territorio y F6
estado del juego sin balon. Todo colectivo, sin identidades, en pares _h/_a (salvo F6, que es del
partido) y con su cambio despues - antes, como tactical.py.

F3 (E058) - lo que ve un entrenador en los 1-3 s tras un robo (Bauer y Anzer 2021, contrapresion):
  el que roba sale hacia la porteria rival (el centroide avanza), abre a los extremos (los dos mas
  abiertos se separan) y suben los laterales; el que pierde contrapresiona (jugadores que esprintan
  hacia el balon o hacia el poseedor probable) o repliega (el centroide retrocede).
F5 (E059) - territorio: altura media de cada equipo en 3, 10 y 30 s, quien empuja a quien (velocidad
  del centroide hacia delante) y "momentum" (diferencia de alturas integrada).
F6 (E071) - estado del juego sin balon (Lang et al. 2022, 92 % de acierto solo con jugadores): la
  cinematica colectiva (velocidad media, |aceleracion| media, % parados, % esprintando, dispersion)
  en el instante y desplazada +-0,4 / 0,8 / 2 / 4 / 20 / 40 s. Pesa sobre todo +-4 s.
"""

from __future__ import annotations

import warnings

import numpy as np

from hackbarca.features.tactical import _span_mean, _team_axes
from hackbarca.io.load_match import MatchTracking

SPRINT = 4.0      # m/s
STAND = 1.0       # m/s
SHIFTS_S = (0.4, 0.8, 2.0, 4.0, 20.0, 40.0)


def _velocity(x: np.ndarray, fps: int, h: int = 5) -> np.ndarray:
    """Derivada centrada en +-h frames (m/s), por columnas."""
    v = np.full(x.shape, np.nan)
    v[h:-h] = (x[2 * h:] - x[:-2 * h]) * fps / (2 * h)
    return v


def _change(series: np.ndarray, w: int) -> np.ndarray:
    return _span_mean(series, 1, w) - _span_mean(series, -w, -1)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def transition_features(match: MatchTracking, bx: np.ndarray, by: np.ndarray, signs: dict) -> dict:
    """F3: firmas del robo y de la contrapresion, por equipo."""
    fps = match.fps
    axes = _team_axes(match, signs)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in ("h", "a"):
            adv, lat = axes[name]
            vadv, vlat = _velocity(adv, fps), _velocity(lat, fps)
            # centroide hacia delante (m/s): el que roba sale, el que pierde repliega
            cen_v = np.nanmean(vadv, axis=1)
            out["tr_cen_v_" + name] = _span_mean(cen_v, -12, 12)
            for w in (25, 75):
                out["tr_cen_v_chg%d_%s" % (w, name)] = _change(cen_v, w)
            # apertura: separacion lateral de los dos mas abiertos (extremos/laterales)
            spread = np.nanmax(lat, axis=1) - np.nanmin(lat, axis=1)
            out["tr_open_chg75_" + name] = _change(spread, 75)
            # laterales: velocidad hacia delante de los dos jugadores mas abiertos
            order = np.argsort(np.abs(np.where(np.isnan(lat), 0, lat)), axis=1)
            rows = np.arange(len(lat))[:, None]
            wide_v = np.nanmean(vadv[rows, order[:, -2:]], axis=1)
            out["tr_wide_v_" + name] = _span_mean(wide_v, -12, 12)
            out["tr_wide_v_chg75_" + name] = _change(wide_v, 75)
            # contrapresion: jugadores que esprintan hacia el balon (con balon visible)
            px, py = match.players_x[:, match.team_of_track == (0 if name == "h" else 1)], \
                match.players_y[:, match.team_of_track == (0 if name == "h" else 1)]
            vx, vy = _velocity(px, fps), _velocity(py, fps)
            dx, dy = bx[:, None] - px, by[:, None] - py
            dist = np.hypot(dx, dy)
            toward = (vx * dx + vy * dy) / np.maximum(dist, 1e-3)
            press = ((toward > SPRINT) & (dist < 20)).sum(1).astype(float)
            press[np.isnan(bx)] = np.nan
            out["tr_press_" + name] = _span_mean(press, -12, 12)
            for w in (25, 75):
                out["tr_press_chg%d_%s" % (w, name)] = _change(press, w)
    return out


def territory_features(match: MatchTracking, signs: dict) -> dict:
    """F5: territorio e inclinacion del campo."""
    fps = match.fps
    axes = _team_axes(match, signs)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        height = {name: np.nanmean(axes[name][0], axis=1) for name in ("h", "a")}
        for name, other in (("h", "a"), ("a", "h")):
            for w in (75, 250, 750):
                out["te_height_m%d_%s" % (w, name)] = _span_mean(height[name], -w, w)
            push = _velocity(height[name][:, None], fps, h=12)[:, 0]
            out["te_push_m75_" + name] = _span_mean(push, -75, 75)
            # inclinacion: altura propia menos la del rival (coordenadas de ataque opuestas: suma)
            tilt = height[name] + height[other]
            out["te_tilt_m250_" + name] = _span_mean(tilt, -250, 250)
            out["te_tilt_chg75_" + name] = _change(tilt, 75)
    return out


def gamestate_features(match: MatchTracking) -> dict:
    """F6: cinematica colectiva multiescala (Lang et al. 2022), del partido (no por equipo)."""
    fps = match.fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vx, vy = _velocity(match.players_x, fps), _velocity(match.players_y, fps)
        speed = np.hypot(vx, vy)
        acc = np.abs(_velocity(speed, fps))
        base = {"gs_speed": np.nanmean(speed, axis=1), "gs_acc": np.nanmean(acc, axis=1),
                "gs_stand": np.nanmean(speed < STAND, axis=1), "gs_sprint": np.nanmean(speed > SPRINT, axis=1),
                "gs_spread": np.nanstd(match.players_x, axis=1) + np.nanstd(match.players_y, axis=1)}
    out = {}
    for key, series in base.items():
        smooth = _span_mean(series, -5, 5)
        out[key] = smooth
        for s in SHIFTS_S:
            k = int(round(s * fps))
            out["%s_p%g" % (key, s)] = _shift(smooth, k)
            out["%s_m%g" % (key, s)] = _shift(smooth, -k)
    return out


# ----------------------------------------------------------------------------- F4 pressing intensity

REACTION = 0.7      # s (Bekkers 2025)
VMAX = 7.0          # m/s
T_PRESS = 1.5       # s: ventana de presion
SIGMA = 0.45


def pressing_features(match: MatchTracking, bx: np.ndarray, by: np.ndarray) -> dict:
    """F4 (E072): presion de cada equipo sobre el otro como probabilidad de interceptar en T_PRESS s.

    Tiempo hasta interceptar de cada defensor a cada atacante: tras REACTION s siguiendo su velocidad,
    linea recta a VMAX. p = sigmoide((T - TTI) * pi / (sqrt(3) sigma)). Presion sobre un atacante =
    1 - prod(1 - p) de los defensores. Por equipo: presion media que recibe y la que recibe su jugador
    mas cercano al balon (el poseedor probable). Quien tiene el balon es quien recibe la presion.
    """
    fps = match.fps
    out = {}
    k = np.pi / (np.sqrt(3) * SIGMA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vx, vy = _velocity(match.players_x, fps), _velocity(match.players_y, fps)
        for att, name in ((0, "h"), (1, "a")):
            a_cols = match.team_of_track == att
            d_cols = ~a_cols
            ax, ay = match.players_x[:, a_cols], match.players_y[:, a_cols]
            dx = match.players_x[:, d_cols] + np.nan_to_num(vx[:, d_cols]) * REACTION
            dy = match.players_y[:, d_cols] + np.nan_to_num(vy[:, d_cols]) * REACTION
            n = match.n_frames
            mean_p = np.full(n, np.nan)
            carrier_p = np.full(n, np.nan)
            for s in range(0, n, 5000):
                e = min(s + 5000, n)
                dist = np.hypot(ax[s:e, :, None] - dx[s:e, None, :], ay[s:e, :, None] - dy[s:e, None, :])
                tti = REACTION + dist / VMAX
                p = 1.0 / (1.0 + np.exp(-k * (T_PRESS - tti)))
                p = np.where(np.isnan(p), 0.0, p)
                press = 1.0 - np.prod(1.0 - p, axis=2)                 # (frames, atacantes)
                press = np.where(np.isnan(ax[s:e]), np.nan, press)
                mean_p[s:e] = np.nanmean(press, axis=1)
                db = np.hypot(ax[s:e] - bx[s:e, None], ay[s:e] - by[s:e, None])
                has = np.isfinite(db).any(1)
                idx = np.argmin(np.where(np.isfinite(db), db, np.inf), axis=1)
                carrier_p[s:e] = np.where(has, press[np.arange(e - s), idx], np.nan)
            out["pi_mean_" + name] = _span_mean(mean_p, -12, 12)
            out["pi_carrier_" + name] = _span_mean(carrier_p, -12, 12)
            for w in (25, 75):
                out["pi_carrier_chg%d_%s" % (w, name)] = _change(carrier_p, w)
    return out


# ----------------------------------------------------------------------------- F1 roles y formacion

def _roles(rel: np.ndarray, iters: int = 3, sample: int = 25):
    """Bialkowski: plantilla de roles (k x 2) por asignacion hungara iterada sobre frames muestreados.
    rel: (frames, slots, 2) posiciones relativas al centroide del equipo (NaN si el slot no esta)."""
    from scipy.optimize import linear_sum_assignment
    frames = [f for f in range(0, len(rel), sample) if np.isfinite(rel[f, :, 0]).sum() >= 9]
    if len(frames) < 20:
        return None
    k = 10
    first = rel[frames[0]][np.isfinite(rel[frames[0], :, 0])]
    order = np.lexsort((first[:, 1], first[:, 0]))
    template = first[order][:k] if len(first) >= k else None
    if template is None:
        return None
    for _ in range(iters):
        acc = np.zeros((k, 2))
        cnt = np.zeros(k)
        for f in frames:
            pts = rel[f][np.isfinite(rel[f, :, 0])]
            cost = np.hypot(template[:, None, 0] - pts[None, :, 0], template[:, None, 1] - pts[None, :, 1])
            r, c = linear_sum_assignment(cost)
            acc[r] += pts[c]
            cnt[r] += 1
        template = acc / np.maximum(cnt, 1)[:, None]
    return template


def _template_distance(rel: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Distancia media (m) de cada frame a la plantilla con la mejor asignacion de roles."""
    from scipy.optimize import linear_sum_assignment
    out = np.full(len(rel), np.nan)
    for f in range(len(rel)):
        pts = rel[f][np.isfinite(rel[f, :, 0])]
        if len(pts) < 6:
            continue
        cost = np.hypot(template[:, None, 0] - pts[None, :, 0], template[:, None, 1] - pts[None, :, 1])
        r, c = linear_sum_assignment(cost)
        out[f] = cost[r, c].mean()
    return out


def formation_features(match: MatchTracking, bx: np.ndarray, by: np.ndarray, signs: dict, step: int = 5) -> dict:
    """F1 (E073): parecido de cada equipo a su formacion con balon y sin balon (Shaw y Glickman),
    con roles de Bialkowski. Sin GT: "con balon" son los frames en que su jugador es el mas cercano a
    un balon visible y "sin balon" los del rival (senal ruidosa, pero sin etiquetas). Se calcula cada
    `step` frames y se interpola; las plantillas, por equipo y parte."""
    axes = _team_axes(match, signs)
    period = match.period if match.period is not None else np.ones(match.n_frames, dtype=np.int8)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = {}
        for name in ("h", "a"):
            adv, lat = axes[name]
            dist[name] = np.nanmin(np.hypot(match.players_x[:, match.team_of_track == (0 if name == "h" else 1)]
                                            - bx[:, None], match.players_y[:, match.team_of_track ==
                                                                            (0 if name == "h" else 1)] - by[:, None]), axis=1)
        for name, other in (("h", "a"), ("a", "h")):
            adv, lat = axes[name]
            # sin el portero (el mas retrasado), relativo al centroide
            rel = np.stack([adv - np.nanmean(adv, axis=1, keepdims=True), lat - np.nanmean(lat, axis=1, keepdims=True)], -1)
            gk = np.argmin(np.where(np.isnan(adv), np.inf, adv), axis=1)
            rel[np.arange(len(rel)), gk] = np.nan
            with_ball = np.isfinite(bx) & (dist[name] < dist[other]) & (dist[name] < 3)
            without = np.isfinite(bx) & (dist[other] < dist[name]) & (dist[other] < 3)
            d_with = np.full(match.n_frames, np.nan)
            d_without = np.full(match.n_frames, np.nan)
            for p in np.unique(period):
                sel = np.flatnonzero(period == p)
                t_with = _roles(rel[sel[with_ball[sel]]]) if with_ball[sel].sum() > 500 else None
                t_without = _roles(rel[sel[without[sel]]]) if without[sel].sum() > 500 else None
                if t_with is None or t_without is None:
                    continue
                sub = sel[::step]
                d_with[sub] = _template_distance(rel[sub], t_with)
                d_without[sub] = _template_distance(rel[sub], t_without)
            idx = np.arange(match.n_frames)
            for arr in (d_with, d_without):
                ok = np.isfinite(arr)
                if ok.sum() > 1:
                    arr[:] = np.interp(idx, idx[ok], arr[ok])
            score = d_without - d_with          # + = se parece mas a su forma con balon
            out["fo_with_" + name] = _span_mean(score, -12, 12)
            for w in (25, 75):
                out["fo_with_chg%d_%s" % (w, name)] = _change(score, w)
    return out


# ----------------------------------------------------------------------------- columnas extra

EXTRA_FAMILIES = ("transition", "gamestate", "pressing", "formation", "territory")


def extra_features(match: MatchTracking, names) -> "pd.DataFrame":
    """Columnas de las familias de entrenador pedidas (se anaden a las de frame_features sin tocar su
    cache: la clave de tune_l3 incluye este fichero y los nombres)."""
    import pandas as pd

    from hackbarca.features.frame_features import _attack_sign
    from hackbarca.io.load_match import AWAY, HOME
    from hackbarca.models.l1 import interpolate_ball
    unknown = set(names) - set(EXTRA_FAMILIES)
    if unknown:
        raise ValueError("familias extra desconocidas: {}".format(sorted(unknown)))
    bx, by, _ = interpolate_ball(match, 10)
    signs = {"h": _attack_sign(match, HOME), "a": _attack_sign(match, AWAY)}
    f = {}
    if "transition" in names:
        f.update(transition_features(match, bx, by, signs))
    if "gamestate" in names:
        f.update(gamestate_features(match))
    if "pressing" in names:
        f.update(pressing_features(match, bx, by))
    if "formation" in names:
        f.update(formation_features(match, bx, by, signs))
    if "territory" in names:
        f.update({k: v for k, v in territory_features(match, signs).items() if "tilt" not in k})
    return pd.DataFrame(f).astype(np.float32)
