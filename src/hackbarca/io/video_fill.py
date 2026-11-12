"""Relleno del balon en los huecos del ATD con el detector de video (E061/E063).

Real: `apply_video_fill` pone en los huecos del ATD el balon que el YOLO afinado encontro en el video
(data/derived/video_ball.npz, `vision.ball_detector`). Banco: `simulate_video_fill` imita ese relleno
sobre un partido degradado, con la forma medida en el reto (solo agregados, `vision.fill_shape`):

- cubre ~56 % de los frames de hueco, en tramos de media 41 frames separados por tramos de 33;
- error en los cubiertos p50 0,9 m, 72 % a < 2 m y cola larga (p90 9,4 m, 10 % a > 10 m): por tramo,
  o sigue al balon con un error pequeno o se engancha a otra cosa con un desvio fijo. El error es
  suave en el tiempo (AR(1)): el enlace real penaliza los saltos, y un ruido independiente por frame
  daria velocidades falsas (E064: ball_speed p50 5,3 m/s en el banco frente a 3,6 en el reto;
  el balon relleno real va a 0,64 m/s de mediana: casi todo es balon parado en paron o reanudacion);
- en los parones el tracking limpio no tiene balon, pero el video si lo ve quieto donde se reanuda:
  ahi el balon simulado es la siguiente posicion limpia conocida (ball_known_m125 quedaba por debajo).

v3 (E069, tras E068): en el reto el balon relleno esta pegado a un jugador (paron: distancia p50 1,9 m,
p25 0,5 m; vivo: p50 0,6 m) y casi todo cae en parones (9:1). El YOLO solo ve el balon lento y junto a
alguien: en juego vivo solo se cubren frames con el balon limpio lento y cerca de un jugador, y en los
parones una parte de los tramos lleva el balon con el jugador mas cercano al punto de reanudacion (quien
va a sacar). Sin esto el modelo aprende que "balon pegado a un jugador" es siempre posesion y en el reto
los parones salen como juego vivo.

Sin la misma simulacion en el banco, el modelo veria en el reto huecos mas cortos que en el
entrenamiento (desajuste de dominio).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hackbarca.io.load_match import MatchTracking


@dataclass(frozen=True)
class VideoFill:
    on_mean: float = 41.0         # frames: tramo cubierto
    off_mean: float = 33.0        # frames: tramo sin cubrir dentro de un hueco
    inlier_frac: float = 0.72     # tramos que siguen al balon de verdad
    inlier_sigma: float = 0.5     # m, error en esos tramos
    outlier_min: float = 2.0      # m: desvio de un tramo enganchado a otra cosa = min + lognormal
    outlier_median: float = 3.0
    outlier_sigma: float = 1.8
    jitter: float = 0.3           # m, error sobre el desvio fijo
    error_tau: float = 1000.0     # frames: casi un desvio fijo por tramo; el relleno real va a 0,64 m/s (p50)
    dead_hold: bool = True        # parones: balon quieto en la siguiente posicion limpia
    # v3 (None/0 = comportamiento v2)
    live_max_speed: float | None = None   # m/s: en juego vivo solo se cubre el balon mas lento que esto...
    live_max_dist: float | None = None    # m: ...y a menos de esto de un jugador
    dead_carry: float = 0.0               # prob. de que un tramo de paron lleve el balon con un jugador
    carry_offset: float = 0.4             # m, balon respecto al jugador que lo lleva
    outlier_on_player: bool = False       # en juego vivo el enganche sigue a otro jugador (su bota)


VIDEO_FILL_V2 = VideoFill()
VIDEO_FILL = VideoFill(inlier_sigma=0.3, live_max_speed=6.0, live_max_dist=0.8, dead_carry=0.3,
                       outlier_on_player=True)


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _smooth_noise(rng, n: int, sigma: float, tau: float) -> np.ndarray:
    """AR(1) estacionario de desviacion `sigma` y escala `tau` frames."""
    from scipy.signal import lfilter
    rho = np.exp(-1.0 / max(tau, 1.0))
    out = lfilter([sigma * np.sqrt(1 - rho ** 2)], [1, -rho], rng.normal(0, 1, n))
    out += (rho ** np.arange(n)) * rng.normal(0, sigma)   # arranque ya en regimen
    return out


def _target(clean: MatchTracking, dead_hold: bool):
    """Balon que veria el video: el limpio y, donde falta (parones), la siguiente posicion conocida."""
    x, y = clean.ball_x.copy(), clean.ball_y.copy()
    if dead_hold:
        known = np.flatnonzero(~np.isnan(x))
        if len(known):
            nxt = known[np.minimum(np.searchsorted(known, np.arange(len(x))), len(known) - 1)]
            x, y = x[nxt], y[nxt]
            after = np.arange(len(x)) > known[-1]
            x[after], y[after] = np.nan, np.nan
    return x, y


def _nearest_player(clean: MatchTracking, x: np.ndarray, y: np.ndarray):
    """Distancia y posicion del jugador limpio mas cercano a (x, y) en cada frame."""
    with np.errstate(invalid="ignore"):
        d = np.hypot(clean.players_x - x[:, None], clean.players_y - y[:, None])
        d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    rows = np.arange(len(x))
    dist = d[rows, k]
    px, py = clean.players_x[rows, k], clean.players_y[rows, k]
    bad = ~np.isfinite(dist)
    px[bad], py[bad] = np.nan, np.nan
    return dist, px, py


def _v3_targets(rng, clean: MatchTracking, target_x, target_y, fill: VideoFill):
    """Que frames puede ver el video y donde: juego vivo lento y junto a alguien; paron quieto o llevado."""
    live = ~np.isnan(clean.ball_x)
    detectable = ~np.isnan(target_x)
    if fill.live_max_speed is not None:
        speed = np.full(len(target_x), np.inf)
        speed[1:] = np.hypot(np.diff(clean.ball_x), np.diff(clean.ball_y)) * clean.fps
        speed = np.where(np.isnan(speed), np.inf, speed)
        dist, _, _ = _nearest_player(clean, clean.ball_x, clean.ball_y)
        detectable &= ~live | ((speed < fill.live_max_speed) & (dist < fill.live_max_dist))
    if fill.dead_carry > 0:
        _, px, py = _nearest_player(clean, target_x, target_y)   # quien esta junto al punto de reanudacion
        for a, b in _runs(~live & detectable):
            if rng.random() < fill.dead_carry:
                angle = rng.uniform(0, 2 * np.pi)
                ok = ~np.isnan(px[a:b])
                target_x[a:b][ok] = px[a:b][ok] + fill.carry_offset * np.cos(angle)
                target_y[a:b][ok] = py[a:b][ok] + fill.carry_offset * np.sin(angle)
    return detectable


def _other_player(rng, clean: MatchTracking, seg: np.ndarray, target_x, target_y, fill: VideoFill):
    """Enganche a la bota de otro jugador: sigue a un jugador a 2-20 m del balon al empezar el tramo."""
    f = seg[0]
    with np.errstate(invalid="ignore"):
        d = np.hypot(clean.players_x[f] - target_x[f], clean.players_y[f] - target_y[f])
    cands = np.flatnonzero((d >= fill.outlier_min) & (d <= 20.0))
    if not len(cands):
        return None
    k = cands[rng.integers(len(cands))]
    angle = rng.uniform(0, 2 * np.pi)
    jitter_x = _smooth_noise(rng, len(seg), fill.jitter, fill.error_tau)
    jitter_y = _smooth_noise(rng, len(seg), fill.jitter, fill.error_tau)
    x = clean.players_x[seg, k] + fill.carry_offset * np.cos(angle) + jitter_x
    y = clean.players_y[seg, k] + fill.carry_offset * np.sin(angle) + jitter_y
    return x, y   # NaN donde ese jugador no esta: el video tampoco daria balon


def simulate_video_fill(degraded: MatchTracking, clean: MatchTracking, fill: VideoFill = VIDEO_FILL,
                        seed: int = 0) -> MatchTracking:
    """Rellena parte de los huecos de `degraded` con el balon limpio mas el error del detector."""
    rng = np.random.default_rng(10_000 + seed)
    ball_x, ball_y = degraded.ball_x.copy(), degraded.ball_y.copy()
    target_x, target_y = _target(clean, fill.dead_hold)
    detectable = ~np.isnan(target_x)
    if fill.live_max_speed is not None or fill.dead_carry > 0:
        detectable = _v3_targets(np.random.default_rng(20_000 + seed), clean, target_x, target_y, fill)
    fillable = np.isnan(ball_x) & detectable
    total = fill.on_mean + fill.off_mean
    for a, b in _runs(np.isnan(ball_x)):
        t = a
        covered = rng.random() < fill.on_mean / total
        while t < b:
            length = int(rng.geometric(1.0 / (fill.on_mean if covered else fill.off_mean)))
            stop = min(t + length, b)
            if covered:
                seg = np.arange(t, stop)
                seg = seg[fillable[seg]]
                if len(seg):
                    m = len(seg)
                    other = None
                    if rng.random() < fill.inlier_frac:
                        x = target_x[seg] + _smooth_noise(rng, m, fill.inlier_sigma, fill.error_tau)
                        y = target_y[seg] + _smooth_noise(rng, m, fill.inlier_sigma, fill.error_tau)
                    elif fill.outlier_on_player and not np.isnan(clean.ball_x[seg[0]]) and (other := _other_player(rng, clean, seg, target_x, target_y,
                                                                            fill)) is not None:
                        x, y = other
                    else:
                        r = fill.outlier_min + np.exp(rng.normal(np.log(fill.outlier_median), fill.outlier_sigma))
                        angle = rng.uniform(0, 2 * np.pi)
                        r = min(r, 40.0)
                        x = target_x[seg] + r * np.cos(angle) + _smooth_noise(rng, m, fill.jitter, fill.error_tau)
                        y = target_y[seg] + r * np.sin(angle) + _smooth_noise(rng, m, fill.jitter, fill.error_tau)
                    ball_x[seg], ball_y[seg] = x, y
            covered = not covered
            t = stop
    return replace(degraded, ball_x=ball_x, ball_y=ball_y)


def apply_video_fill(match: MatchTracking, path: str) -> MatchTracking:
    """Pone en los frames sin balon del ATD el balon detectado en el video (NaN donde no lo hay)."""
    blob = np.load(path)
    frames, x, y = blob["frame"].astype(int), blob["x"], blob["y"]
    ball_x, ball_y = match.ball_x.copy(), match.ball_y.copy()
    keep = (frames < match.n_frames) & ~np.isnan(x)
    keep[keep] &= np.isnan(ball_x[frames[keep]])
    ball_x[frames[keep]] = x[keep]
    ball_y[frames[keep]] = y[keep]
    return replace(match, ball_x=ball_x, ball_y=ball_y)
