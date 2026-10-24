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


def simulate_video_fill(degraded: MatchTracking, clean: MatchTracking, fill: VideoFill = VideoFill(),
                        seed: int = 0) -> MatchTracking:
    """Rellena parte de los huecos de `degraded` con el balon limpio mas el error del detector."""
    rng = np.random.default_rng(10_000 + seed)
    ball_x, ball_y = degraded.ball_x.copy(), degraded.ball_y.copy()
    target_x, target_y = _target(clean, fill.dead_hold)
    fillable = np.isnan(ball_x) & ~np.isnan(target_x)
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
                    if rng.random() < fill.inlier_frac:
                        dx = _smooth_noise(rng, m, fill.inlier_sigma, fill.error_tau)
                        dy = _smooth_noise(rng, m, fill.inlier_sigma, fill.error_tau)
                    else:
                        r = fill.outlier_min + np.exp(rng.normal(np.log(fill.outlier_median), fill.outlier_sigma))
                        angle = rng.uniform(0, 2 * np.pi)
                        r = min(r, 40.0)
                        dx = r * np.cos(angle) + _smooth_noise(rng, m, fill.jitter, fill.error_tau)
                        dy = r * np.sin(angle) + _smooth_noise(rng, m, fill.jitter, fill.error_tau)
                    ball_x[seg] = target_x[seg] + dx
                    ball_y[seg] = target_y[seg] + dy
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
