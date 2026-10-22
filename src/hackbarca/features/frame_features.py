"""Features por frame para L3, identicas en el ATD y en el banco externo.

Sin identidades (solo equipo por slot) y sin depender de hacia donde ataca cada equipo: la
direccion de ataque se estima por mitad con la posicion media de cada equipo. Las features de
equipo van en pares (home, away) para poder intercambiar equipos al aumentar datos (`swap_teams`).
Todo es offline: se usan ventanas centradas y el contexto del hueco entero sin balon.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from hackbarca.io.load_match import AWAY, HOME, MatchTracking
from hackbarca.models.l1 import _gap_lengths, interpolate_ball

WINDOWS = (12, 50, 125)


def _centered_mean(values: np.ndarray, half: int) -> np.ndarray:
    """Media centrada en +-half frames ignorando NaN."""
    ok = ~np.isnan(values)
    v = np.where(ok, values, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(values))
    lo, hi = np.maximum(0, idx - half), np.minimum(len(values), idx + half + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def _team_speed(match: MatchTracking, team: int, lag: int) -> np.ndarray:
    cols = match.team_of_track == team
    x, y = match.players_x[:, cols], match.players_y[:, cols]
    out = np.full(match.n_frames, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.sqrt((x[lag:] - x[:-lag]) ** 2 + (y[lag:] - y[:-lag]) ** 2) * (match.fps / lag)
        out[lag // 2:lag // 2 + len(v)] = np.nanmedian(v, axis=1)
    return out


def _attack_sign(match: MatchTracking, team: int) -> np.ndarray:
    """+1 si el equipo ataca hacia x creciente en ese frame (estimado por mitad), -1 si no."""
    period = match.period if match.period is not None else np.ones(match.n_frames, dtype=np.int8)
    cols = match.team_of_track == team
    sign = np.ones(match.n_frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_x = np.nanmean(match.players_x[:, cols], axis=1)
    for p in np.unique(period):
        sel = period == p
        # un equipo defiende su porteria: su media cae en su propio campo
        sign[sel] = 1.0 if np.nanmean(mean_x[sel]) < match.pitch_length / 2 else -1.0
    return sign


def _gap_context(match: MatchTracking, team_near: np.ndarray):
    """Para cada frame sin balon: equipo mas cercano justo antes y justo despues del hueco."""
    missing = np.isnan(match.ball_x)
    before = np.full(match.n_frames, np.nan)
    after = np.full(match.n_frames, np.nan)
    since = np.zeros(match.n_frames)
    until = np.zeros(match.n_frames)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if a > 0:
            before[a:b] = team_near[a - 1]
        if b < match.n_frames:
            after[a:b] = team_near[b]
        since[a:b] = np.arange(1, b - a + 1)
        until[a:b] = np.arange(b - a, 0, -1)
    return before, after, since, until


TOUCH_DV = 9.0  # m/s de cambio de velocidad del balon para considerar un toque
TOUCH_W = 8     # frames a cada lado para estimar la velocidad antes y despues
TOUCH_RATE = 0.4  # v3: toques por segundo de balon visible (eventos con balon en Metrica: ~0,42)


def _centered_max(values: np.ndarray, half: int) -> np.ndarray:
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return windows.max(axis=1)


def _fill_last(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Ultimo valor valido hasta cada frame (NaN antes del primero)."""
    idx = np.where(valid, np.arange(len(values)), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)


def _v2_features(match: MatchTracking, bx, by, known, dist, team_near, relative: bool = False) -> dict:
    """Toques, receptor, cinematica del mas cercano, presion y compacidad (features v2)."""
    n, fps = match.n_frames, match.fps
    f = {}
    # toque = cambio brusco entre la velocidad media de los TOUCH_W frames anteriores y la de los
    # posteriores; con ventanas cortas el ruido del tracker (~0,3 m) dispara toques falsos
    w = TOUCH_W
    before_x, before_y = np.full(n, np.nan), np.full(n, np.nan)
    after_x, after_y = np.full(n, np.nan), np.full(n, np.nan)
    before_x[w:] = (bx[w:] - bx[:-w]) * fps / w
    before_y[w:] = (by[w:] - by[:-w]) * fps / w
    after_x[:-w] = (bx[w:] - bx[:-w]) * fps / w
    after_y[:-w] = (by[w:] - by[:-w]) * fps / w
    dv = np.hypot(after_x - before_x, after_y - before_y)
    vx, vy = after_x, after_y
    dvz = np.nan_to_num(dv)
    local_max = dvz >= _centered_max(dvz, 5)
    if relative:
        # v3: umbral por partido para TOUCH_RATE toques/s de balon visible (sin etiquetas): la
        # feature significa lo mismo aunque el tracker suavice mas o menos las patadas
        candidates = np.sort(dvz[local_max & known & (dvz > 0)])[::-1]
        k = int(TOUCH_RATE * known.sum() / fps)
        threshold = candidates[min(k, len(candidates) - 1)] if len(candidates) else np.inf
        touch = (dvz > threshold) & local_max
        ranks = np.full(n, np.nan)
        ok = ~np.isnan(dv)
        ranks[ok] = (np.argsort(np.argsort(dv[ok])) + 0.5) / ok.sum()
        f["touch_rank"] = ranks
    else:
        touch = (dvz > TOUCH_DV) & local_max
        f["touch_dv"] = dv
    sign = np.nan_to_num(team_near)
    f["last_touch_sign"] = _fill_last(sign, touch)
    f["next_touch_sign"] = _fill_last(sign[::-1], touch[::-1])[::-1]
    idx = np.arange(n)
    last_idx = _fill_last(idx.astype(float), touch)
    next_idx = _fill_last(idx[::-1].astype(float), touch[::-1])[::-1]
    f["since_touch_log"] = np.log1p(np.where(np.isnan(last_idx), 1e4, idx - last_idx))
    f["to_touch_log"] = np.log1p(np.where(np.isnan(next_idx), 1e4, next_idx - idx))
    signed_touch = np.where(touch, sign, 0.0)
    for w in WINDOWS[:2]:
        f["touch_sign_m%d" % w] = _centered_mean(signed_touch, w)

    moving = known & (np.hypot(np.nan_to_num(vx), np.nan_to_num(vy)) > 5.0)
    for team, name in ((HOME, "h"), (AWAY, "a")):
        cols = match.team_of_track == team
        px, py = match.players_x[:, cols], match.players_y[:, cols]
        d = dist[:, cols]
        # receptor: distancia del jugador mas cercano a donde ira el balon en 0,5 s y 1 s
        for tau, tag in ((0.5, "05"), (1.0, "10")):
            fx = np.nan_to_num(bx) + np.nan_to_num(vx) * tau
            fy = np.nan_to_num(by) + np.nan_to_num(vy) * tau
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = np.nanmin(np.hypot(px - fx[:, None], py - fy[:, None]), axis=1)
            f["recv%s_%s" % (tag, name)] = np.where(moving, r, np.nan)
        # el mas cercano: velocidad de acercamiento al balon y presion rival a su alrededor
        has = np.isfinite(d).any(1)
        k = np.argmin(d, axis=1)
        d1 = np.where(has, d[idx, k], np.nan)
        close = np.full(n, np.nan)
        close[2:-2] = -(d1[4:] - d1[:-4]) * fps / 4
        f["close_" + name] = close
        nx, ny = px[idx, k], py[idx, k]
        other = match.team_of_track != team
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            od = np.hypot(match.players_x[:, other] - nx[:, None], match.players_y[:, other] - ny[:, None])
            f["press_" + name] = np.where(has & known, (od < 5).sum(1), np.nan).astype(float)
            f["spread_x_" + name] = np.nanstd(px, axis=1)
            f["spread_y_" + name] = np.nanstd(py, axis=1)
    return f


def _v4_features(match: MatchTracking, f: dict) -> dict:
    """Movimiento colectivo para los huecos sin balon: quien empuja y quien repliega."""
    n, fps = match.n_frames, match.fps
    out = {}
    for name in ("h", "a"):
        for base in ("block", "front"):
            series = f["%s_%s" % (base, name)]
            v = np.full(n, np.nan)
            v[fps // 2:fps // 2 + n - fps] = series[fps:] - series[:-fps]  # m/s, diferencia en 1 s
            for w in WINDOWS[1:]:
                out["%s_v_m%d_%s" % (base, w, name)] = _centered_mean(v, w)
    missing = np.isnan(match.ball_x)
    jump = np.full(n, np.nan)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if a > 0 and b < n:
            jump[a:b] = np.hypot(match.ball_x[b] - match.ball_x[a - 1], match.ball_y[b] - match.ball_y[a - 1])
    out["gap_jump"] = jump
    return out


SIGNED_V2 = ("last_touch_sign", "next_touch_sign") + tuple("touch_sign_m%d" % w for w in WINDOWS[:2])

# Familias de features (plan de investigacion). "base" siempre va; "touch" son los toques con umbral
# relativo al partido, receptor, presion y dispersion (antes v2/v3); el resto vive en tactical.py
# (balon, jugadores, bloques, estructura) y en _layout_features (reanudacion).
FAMILIES = ("base", "touch", "ball", "players", "blocks", "structure", "restart")
# alias de las versiones numeradas antiguas: 3 es el modelo vigente (ADR-0011) y no cambia
VERSION_GROUPS = {
    1: ("base",), 2: ("base", "touch_fixed"), 3: ("base", "touch"), 4: ("base", "touch", "v4"),
}


def resolve_groups(version) -> tuple:
    """Familias de una version: entero antiguo (1-4) o nombres unidos por '+' ("base+touch+ball")."""
    if isinstance(version, (int, np.integer)):
        return VERSION_GROUPS[int(version)]
    names = tuple(str(version).split("+")) if isinstance(version, str) else tuple(version)
    unknown = set(names) - set(FAMILIES) - {"touch_fixed", "v4"}
    if unknown:
        raise ValueError("familias desconocidas: {}".format(sorted(unknown)))
    return tuple(dict.fromkeys(("base",) + names))


def frame_features(match: MatchTracking, windows=None, max_gap: int = 10,
                   version=1) -> pd.DataFrame:
    """Features por frame. `version`: entero antiguo (3 = modelo vigente) o familias unidas por
    '+', p. ej. "touch+ball+players+blocks+structure+restart" (ver FAMILIES)."""
    n = match.n_frames
    raw_missing = np.isnan(match.ball_x)
    bx, by, _ = interpolate_ball(match, max_gap)
    known = ~np.isnan(bx)
    gap = _gap_lengths(raw_missing)

    dx = match.players_x - np.nan_to_num(bx)[:, None]
    dy = match.players_y - np.nan_to_num(by)[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.where(np.isnan(dist) | ~known[:, None], np.inf, dist)

    f = {}
    f["ball_known"] = known.astype(float)
    f["gap_log"] = np.log1p(gap)
    f["visible_players"] = (~np.isnan(match.players_x)).sum(1).astype(float)
    edge = np.minimum.reduce([bx, match.pitch_length - bx, by, match.pitch_width - by])
    f["ball_edge_dist"] = np.where(known, edge, np.nan)
    f["ball_goal_dist"] = np.where(known, np.minimum(bx, match.pitch_length - bx), np.nan)
    speed = np.full(n, np.nan)
    speed[2:-2] = np.hypot(bx[4:] - bx[:-4], by[4:] - by[:-4]) * (match.fps / 4)
    f["ball_speed"] = speed

    nearest = {}
    for team, name in ((HOME, "h"), (AWAY, "a")):
        d = dist[:, match.team_of_track == team]
        d_sorted = np.sort(d, axis=1)
        d1 = np.where(np.isfinite(d_sorted[:, 0]), d_sorted[:, 0], np.nan)
        d2 = np.where(np.isfinite(d_sorted[:, 1]), d_sorted[:, 1], np.nan)
        nearest[team] = d1
        f["d1_" + name] = d1
        f["d2_" + name] = d2
        f["n3_" + name] = np.where(known, (d < 3).sum(1), np.nan).astype(float)
        f["n8_" + name] = np.where(known, (d < 8).sum(1), np.nan).astype(float)
        f["speed_" + name] = _team_speed(match, team, match.fps)
        sign = _attack_sign(match, team)
        cols = match.team_of_track == team
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xs = match.players_x[:, cols]
            adv = (xs - match.pitch_length / 2) * sign[:, None]
            f["block_" + name] = np.nanmedian(adv, axis=1)
            f["front_" + name] = np.nanmax(adv, axis=1)
            f["ball_adv_" + name] = np.where(known, (bx - match.pitch_length / 2) * sign, np.nan)

    team_near = np.where(np.nan_to_num(nearest[HOME], nan=np.inf) <= np.nan_to_num(nearest[AWAY], nan=np.inf),
                         1.0, -1.0)
    team_near[~known] = np.nan
    f["near_sign"] = team_near
    before, after, since, until = _gap_context(match, np.nan_to_num(team_near))
    f["gap_before_sign"] = before
    f["gap_after_sign"] = after
    f["gap_since_log"] = np.log1p(since)
    f["gap_until_log"] = np.log1p(until)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coll = np.nanmean(np.stack([f["speed_h"], f["speed_a"]]), axis=0)
    f["speed_all"] = coll
    for w in WINDOWS:
        f["near_sign_m%d" % w] = _centered_mean(team_near, w)
        f["speed_all_m%d" % w] = _centered_mean(coll, w)
        f["ball_known_m%d" % w] = _centered_mean(f["ball_known"], w)
        f["ddiff_m%d" % w] = _centered_mean(f["d1_a"] - f["d1_h"], w)
        f["edge_m%d" % w] = _centered_mean(f["ball_edge_dist"], w)

    to_tag = np.full(n, 1e4)
    from_tag = np.full(n, 1e4)
    in_tag = np.zeros(n)
    for start, stop in windows or []:
        lo, hi = max(0, start), min(n, stop + 1)
        in_tag[lo:hi] = 1
        idx = np.arange(n)
        to_tag = np.where(idx < start, np.minimum(to_tag, start - idx), to_tag)
        from_tag = np.where(idx > stop, np.minimum(from_tag, idx - stop), from_tag)
    f["tag_in"] = in_tag
    f["tag_to_log"] = np.log1p(to_tag)
    f["tag_from_log"] = np.log1p(from_tag)
    groups = resolve_groups(version)
    if "touch" in groups or "touch_fixed" in groups:
        f.update(_v2_features(match, bx, by, known, dist, team_near, relative="touch" in groups))
    if "v4" in groups:  # rechazado (E015); se conserva para reproducirlo
        f.update(_v4_features(match, f))
    if {"ball", "players", "blocks", "structure"} & set(groups):
        from hackbarca.features import tactical
        signs = {"h": _attack_sign(match, HOME), "a": _attack_sign(match, AWAY)}
        if "ball" in groups:
            f.update(tactical.ball_features(match, bx, by, signs["h"]))
        if "players" in groups:
            f.update(tactical.players_features(match, bx, by, signs))
        if "blocks" in groups or "structure" in groups:
            desc = tactical.block_descriptors(match, signs)
            if "blocks" in groups:
                f.update(tactical.blocks_features(match, desc))
            if "structure" in groups:
                f.update(tactical.structure_features(match, bx, desc, signs))
    if "restart" in groups:
        f.update(_layout_features(match))
    return pd.DataFrame(f).astype(np.float32)  # la mitad de memoria; LightGBM bina igual


BOX_DEPTH, BOX_HALF = 16.5, 20.16    # area grande (m)
SIX_DEPTH, SIX_HALF = 5.5, 9.16      # area pequena
STILL = 1.0                          # m/s: jugador casi parado


def _layout_features(match: MatchTracking) -> dict:
    """Colocacion tipica de balon parado, sin balon (E029): sirve dentro de los huecos del ATD.

    Jugadores quietos, gente amontonada en un area (corner, falta lateral), alguien pegado a la
    banda (saque de banda), alguien en el area pequena (saque de puerta), barrera (grupo compacto)
    y la formacion del saque de centro (cada equipo en su campo y alguien en el centro).
    """
    n, fps = match.n_frames, match.fps
    length, width = match.pitch_length, match.pitch_width
    px, py = match.players_x, match.players_y
    seen = ~np.isnan(px)
    h = fps // 2
    v = np.full(px.shape, np.nan)
    v[h:-h] = np.hypot(px[2 * h:] - px[:-2 * h], py[2 * h:] - py[:-2 * h]) * fps / (2 * h)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        still = np.where(np.isnan(v), np.nan, v < STILL)
        out["frac_still"] = np.nanmean(still, axis=1)
        central = np.abs(py - width / 2)
        out["box_l"] = ((px < BOX_DEPTH) & (central < BOX_HALF)).sum(1).astype(float)
        out["box_r"] = ((px > length - BOX_DEPTH) & (central < BOX_HALF)).sum(1).astype(float)
        out["six_n"] = (((px < SIX_DEPTH) | (px > length - SIX_DEPTH)) & (central < SIX_HALF)).sum(1).astype(float)
        touch = np.minimum(py, width - py)
        out["touch_min"] = np.nanmin(np.where(seen, touch, np.inf), axis=1)
        out["touch_min"][~np.isfinite(out["touch_min"])] = np.nan
        out["center_d"] = np.nanmin(np.hypot(px - length / 2, py - width / 2), axis=1)
        for team, name in ((HOME, "h"), (AWAY, "a")):
            cols = match.team_of_track == team
            sign = _attack_sign(match, team)
            own = (px[:, cols] - length / 2) * sign[:, None] < 0
            out["own_half_" + name] = own.sum(1) / np.maximum(seen[:, cols].sum(1), 1)
    # barrera / grupo compacto: maximo de jugadores a < 2 m de uno mismo
    cluster = np.zeros(n)
    for a in range(0, n, 8192):
        x, y = px[a:a + 8192], py[a:a + 8192]
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        cluster[a:a + len(x)] = np.nanmax(np.where(np.isnan(d), 0, d < 2.0).sum(-1), axis=1, initial=0)
    out["cluster_max"] = cluster  # (n_seen se quito: duplica visible_players y desajustaba, E029)
    for key in ("frac_still", "cluster_max", "box_l", "box_r", "touch_min"):
        for w in (25, 75):
            out["%s_m%d" % (key, w)] = _centered_mean(out[key], w)
    return out


SIGNED = ("near_sign", "gap_before_sign", "gap_after_sign") + tuple(
    "near_sign_m%d" % w for w in WINDOWS)
NEGATED = tuple("ddiff_m%d" % w for w in WINDOWS)


def swap_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Mismo partido con los equipos intercambiados (para aumentar datos y quitar sesgo de local)."""
    out = df.copy()
    for col in df.columns:
        if col.endswith("_h"):
            other = col[:-2] + "_a"
            out[col], out[other] = df[other], df[col]
    from hackbarca.features.tactical import SIGNED_BALL
    for col in SIGNED + NEGATED + SIGNED_V2 + SIGNED_BALL:
        if col in df.columns:
            out[col] = -df[col]
    return out
