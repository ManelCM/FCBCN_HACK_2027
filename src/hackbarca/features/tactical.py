"""Features por familias, pensadas como las mira un entrenador (plan de investigacion, E031).

    balon      -> estado del balon: sentido de progresion, vuelo largo, quieto
    jugadores  -> quien aprieta, quien esprinta, quien se desmarca, el portero
    bloques    -> como de juntos y abiertos estan los equipos, altura de lineas, distancia entre ellos
    estructura -> jugadores por detras del balon, adelantados, forma propia con/sin balon, cambios

Todo sale de posiciones sin identidad, asi que funciona dentro de los huecos sin balon. Las de
equipo van en pares _h/_a (sufijo al final) para `swap_teams`; las del balon vistas desde el local
estan en SIGNED_BALL y se niegan al intercambiar equipos. Cada senal lleva su cambio
despues-menos-antes en 1 s y 3 s: un robo se ve en el cambio, no en el nivel.
"""

from __future__ import annotations

import warnings

import numpy as np

from hackbarca.io.load_match import AWAY, HOME, MatchTracking

SPRINT = 5.0          # m/s
PRESS_RADIUS = 15.0   # m alrededor del balon
PRESS_SPEED = 3.0     # m/s de acercamiento al balon
GK_MAX = 25.0         # m: el mas retrasado solo cuenta como portero si esta cerca de su porteria
FLIGHT_SPEED = 12.0   # m/s: balon en vuelo largo
CHG = (25, 75)        # ventanas antes/despues (1 s y 3 s)
SIGNED_BALL = tuple("ball_prog_%s%d" % (k, w) for k in ("b", "f", "chg") for w in CHG)


def _span_mean(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Media de values en [t+lo, t+hi] ignorando NaN (recortado a los bordes)."""
    ok = ~np.isnan(values)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(values))
    a, b = np.clip(idx + lo, 0, len(values)), np.clip(idx + hi + 1, 0, len(values))
    count = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[b] - cs[a]) / count, np.nan)


def _rate(series: np.ndarray, h: int, fps: int) -> np.ndarray:
    """Derivada centrada en +-h frames (unidades por segundo), por columnas si es 2D."""
    out = np.full(series.shape, np.nan)
    out[h:-h] = (series[2 * h:] - series[:-2 * h]) * fps / (2 * h)
    return out


def _with_changes(out: dict, base: str, team: str, series: np.ndarray) -> None:
    """La serie suavizada en 1 s y su cambio despues-menos-antes en 1 s y 3 s (`..._chg25_h`)."""
    suffix = "_" + team if team else ""
    out[base + suffix] = _span_mean(series, -12, 12)
    for w in CHG:
        out["%s_chg%d%s" % (base, w, suffix)] = _span_mean(series, 1, w) - _span_mean(series, -w, -1)


def _team_axes(match: MatchTracking, signs: dict):
    """Por equipo: coordenada de ataque (propia porteria en -L/2) y lateral de sus slots."""
    half = match.pitch_length / 2
    out = {}
    for team, name in ((HOME, "h"), (AWAY, "a")):
        cols = match.team_of_track == team
        out[name] = ((match.players_x[:, cols] - half) * signs[name][:, None], match.players_y[:, cols])
    return out


# ----------------------------------------------------------------------------------------- balon

def ball_features(match: MatchTracking, bx, by, sign_h) -> dict:
    fps = match.fps
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        known = ~np.isnan(bx)
        vx, vy = _rate(bx, 2, fps), _rate(by, 2, fps)
        prog = vx * sign_h          # + si el balon avanza hacia la porteria que ataca el local
        for w in CHG:
            before, after = _span_mean(prog, -w, -1), _span_mean(prog, 1, w)
            out["ball_prog_b%d" % w] = before
            out["ball_prog_f%d" % w] = after
            out["ball_prog_chg%d" % w] = after - before
        speed = np.hypot(vx, vy)
        d = np.hypot(match.players_x - np.nan_to_num(bx)[:, None], match.players_y - np.nan_to_num(by)[:, None])
        dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        dmin = np.where(known & np.isfinite(dmin), dmin, np.nan)
        out["ball_free_dist"] = dmin
        # vuelo largo: rapido y lejos de todos -> la posesion sigue con quien lo envio
        flight = np.where(known, ((speed > FLIGHT_SPEED) & (dmin > 3.0)).astype(float), np.nan)
        out["ball_flight_m12"] = _span_mean(flight, -12, 12)
        still = np.where(known, (speed < 0.5).astype(float), np.nan)
        out["ball_still_m25"] = _span_mean(still, -25, 25)
        run = np.zeros(match.n_frames)
        s = np.nan_to_num(still) > 0
        for i in range(1, match.n_frames):
            run[i] = run[i - 1] + 1 if s[i] else 0
        out["ball_still_run_log"] = np.log1p(run)
    return out


# ------------------------------------------------------------------------------------- jugadores

def players_features(match: MatchTracking, bx, by, signs: dict) -> dict:
    n, fps = match.n_frames, match.fps
    h = fps // 2
    px, py = match.players_x, match.players_y
    axes = _team_axes(match, signs)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.hypot(_rate(px, h, fps), _rate(py, h, fps))
        known = ~np.isnan(bx)
        for team, other, name in ((HOME, AWAY, "h"), (AWAY, HOME, "a")):
            cols = match.team_of_track == team
            ocols = match.team_of_track == other
            tx, ty = px[:, cols], py[:, cols]
            # sprints: contraataque o contrapresion
            _with_changes(out, "sprint_n", name, (speed[:, cols] > SPRINT).sum(1).astype(float))
            # quien aprieta: jugadores cerca del balon que se acercan rapido, y el mas rapido
            dball = np.hypot(tx - np.nan_to_num(bx)[:, None], ty - np.nan_to_num(by)[:, None])
            approach = -_rate(dball, h, fps)
            pressing = ((dball < PRESS_RADIUS) & (approach > PRESS_SPEED)).sum(1).astype(float)
            pressing[~known] = np.nan
            _with_changes(out, "press_run", name, pressing)
            out["press_max_" + name] = np.where(
                known, np.nanmax(np.where(dball < PRESS_RADIUS, approach, np.nan), axis=1), np.nan)
            # rivales a menos de 5 m del balon (el poseedor suele tener 1-2 encima)
            near = (dball < 5.0).sum(1).astype(float)
            near[~known] = np.nan
            out["near5_" + name] = _span_mean(near, -6, 6)
            # desmarques: ritmo al que los 3 que mas se separan se alejan de su rival mas cercano
            sep = np.full(n, np.nan)
            for a in range(0, n, 8192):
                sl = slice(max(a - 2 * h, 0), min(a + 8192 + 2 * h, n))
                d = np.hypot(tx[sl][:, :, None] - px[sl][:, None, ocols], ty[sl][:, :, None] - py[sl][:, None, ocols])
                dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2)
                dmin[~np.isfinite(dmin)] = np.nan
                rate = _rate(dmin, h, fps)
                top = -np.sort(np.where(np.isnan(rate), np.inf, -rate), axis=1)[:, :3]
                top[~np.isfinite(top)] = np.nan
                vals = np.nanmean(top, axis=1)
                lo = a - sl.start
                sep[a:a + 8192] = vals[lo:lo + min(8192, n - a)]
            _with_changes(out, "unmark", name, sep)
            # portero: el mas retrasado, solo si esta cerca de su porteria (si no, NaN)
            adv, _ = axes[name]
            deepest = np.nanmin(np.where(np.isnan(adv), np.inf, adv), axis=1)
            gk_goal = deepest + match.pitch_length / 2
            gk_goal = np.where(np.isfinite(gk_goal) & (gk_goal < GK_MAX), gk_goal, np.nan)
            _with_changes(out, "gk_goal", name, gk_goal)
    return out


# --------------------------------------------------------------------------------------- bloques

def _hull_area(x: np.ndarray, y: np.ndarray) -> float:
    """Area de la envolvente convexa (cadena monotona de Andrew), NaN con menos de 3 puntos."""
    ok = ~(np.isnan(x) | np.isnan(y))
    pts = sorted(set(zip(x[ok].round(2), y[ok].round(2))))
    if len(pts) < 3:
        return np.nan

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    hx, hy = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(hx, np.roll(hy, 1)) - np.dot(hy, np.roll(hx, 1)))


def block_descriptors(match: MatchTracking, signs: dict) -> dict:
    """Por equipo y frame: longitud, anchura, superficie, stretch index, lineas y centroide, sin el
    portero (el mas retrasado). Son la base de las familias bloques y estructura."""
    axes = _team_axes(match, signs)
    desc = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in ("h", "a"):
            adv, lat = axes[name]
            order = np.argsort(np.where(np.isnan(adv), np.inf, adv), axis=1)
            rows = np.arange(len(adv))[:, None]
            adv_s, lat_s = adv[rows, order], lat[rows, order]
            field_adv, field_lat = adv_s[:, 1:], lat_s[:, 1:]       # sin el portero
            back = np.nanmean(field_adv[:, :4], axis=1)              # linea defensiva
            front = np.nanmax(field_adv, axis=1)
            cx, cy = np.nanmean(field_adv, axis=1), np.nanmean(field_lat, axis=1)
            stretch = np.nanmean(np.hypot(field_adv - cx[:, None], field_lat - cy[:, None]), axis=1)
            length = front - np.nanmin(field_adv, axis=1)
            width = np.nanmax(field_lat, axis=1) - np.nanmin(field_lat, axis=1)
            area = np.array([_hull_area(field_adv[i], field_lat[i]) for i in range(len(adv))])
            desc[name] = dict(length=length, width=width, area=area, stretch=stretch, back=back,
                              front=front, cx=cx, lines_gap=front - back)
    return desc


def blocks_features(match: MatchTracking, desc: dict) -> dict:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in ("h", "a"):
            d = desc[name]
            for key in ("length", "width", "area", "stretch", "back", "front", "lines_gap"):
                _with_changes(out, "blk_" + key, name, d[key])
            out["blk_lpw_" + name] = _span_mean(d["length"] / np.maximum(d["width"], 1.0), -12, 12)
        # distancia entre los dos bloques a lo largo del campo, desde cada equipo (+ si va por delante)
        gap_h = desc["h"]["cx"] + desc["a"]["cx"]   # en coordenadas de ataque opuestas: suma = adelanto del local
        _with_changes(out, "blk_ahead", "h", gap_h)
        _with_changes(out, "blk_ahead", "a", -gap_h)
    return out


# ------------------------------------------------------------------------------------ estructura

def _two_means_score(z: np.ndarray, width_col: int) -> np.ndarray:
    """2-medias deterministas sobre descriptores estandarizados; devuelve d(compacto) - d(abierto).

    Sin etiquetas: el grupo mas ancho es la forma "con balon" del propio equipo en ese partido
    (con balon los equipos son mas anchos y extensos). Inicializado en los cuantiles de anchura.
    """
    ok = ~np.isnan(z).any(1)
    score = np.full(len(z), np.nan)
    if ok.sum() < 50:
        return score
    zz = z[ok]
    w = zz[:, width_col]
    centers = np.vstack([zz[w <= np.quantile(w, 0.25)].mean(0), zz[w >= np.quantile(w, 0.75)].mean(0)])
    for _ in range(20):
        lab = np.argmin(((zz[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.vstack([zz[lab == k].mean(0) if (lab == k).any() else centers[k] for k in range(2)])
        if np.allclose(new, centers):
            break
        centers = new
    if centers[0, width_col] > centers[1, width_col]:
        centers = centers[::-1]
    dist = np.sqrt(((z[ok][:, None, :] - centers[None]) ** 2).sum(-1))
    score[ok] = dist[:, 0] - dist[:, 1]      # + = se parece mas a su forma abierta (con balon)
    return score


def structure_features(match: MatchTracking, bx, desc: dict, signs: dict) -> dict:
    fps = match.fps
    axes = _team_axes(match, signs)
    half = match.pitch_length / 2
    period = match.period if match.period is not None else np.ones(match.n_frames, dtype=np.int8)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        known = ~np.isnan(bx)
        for name in ("h", "a"):
            adv, _ = axes[name]
            ball_adv = (bx - half) * signs[name]
            seen = (~np.isnan(adv)).sum(1)
            behind = np.where(known, (adv < ball_adv[:, None]).sum(1) / np.maximum(seen, 1), np.nan)
            _with_changes(out, "behind_ball", name, behind)
            ahead = (adv > 0).sum(1).astype(float)               # jugadores en campo rival
            _with_changes(out, "in_opp_half", name, ahead)
            # forma propia del partido: descriptores estandarizados por parte
            d = desc[name]
            z = np.column_stack([d[k] for k in ("length", "width", "area", "stretch", "back", "front")])
            shape = np.full(match.n_frames, np.nan)
            zs = np.full_like(z, np.nan)
            for p in np.unique(period):
                sel = period == p
                mu, sd = np.nanmean(z[sel], 0), np.nanstd(z[sel], 0) + 1e-6
                zs[sel] = (z[sel] - mu) / sd
                shape[sel] = _two_means_score(zs[sel], width_col=1)
            _with_changes(out, "shape_open", name, shape)
            # cambio de formacion: cuanto se ha movido la forma en 3 s
            lag = 3 * fps
            moved = np.full(match.n_frames, np.nan)
            moved[lag:] = np.sqrt(np.nansum((zs[lag:] - zs[:-lag]) ** 2, axis=1))
            out["shape_moved_" + name] = _span_mean(moved, -12, 12)
    return out
