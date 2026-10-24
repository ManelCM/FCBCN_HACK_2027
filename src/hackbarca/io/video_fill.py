"""Relleno del balon en los huecos del ATD con el detector de video (E061/E063).

Real: `apply_video_fill` pone en los huecos del ATD el balon que el YOLO afinado encontro en el video
(data/derived/video_ball.npz, `vision.ball_detector`). Banco: `simulate_video_fill` imita ese relleno
sobre un partido degradado, con la forma medida en el reto (solo agregados, `vision.fill_shape`):

- cubre ~56 % de los frames de hueco, en tramos de media 41 frames separados por tramos de 33;
- error en los cubiertos p50 0,9 m, 72 % a < 2 m y cola larga (p90 9,4 m, 10 % a > 10 m): por tramo,
  o sigue al balon con ruido pequeno o se engancha a otra cosa con un desvio fijo.

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
    inlier_sigma: float = 0.5     # m, ruido por frame en esos tramos
    outlier_min: float = 2.0      # m: desvio de un tramo enganchado a otra cosa = min + lognormal
    outlier_median: float = 3.0
    outlier_sigma: float = 1.8
    jitter: float = 0.3           # m, ruido por frame sobre el desvio fijo


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def simulate_video_fill(degraded: MatchTracking, clean: MatchTracking, fill: VideoFill = VideoFill(),
                        seed: int = 0) -> MatchTracking:
    """Rellena parte de los huecos de `degraded` con el balon limpio mas el error del detector."""
    rng = np.random.default_rng(10_000 + seed)
    ball_x, ball_y = degraded.ball_x.copy(), degraded.ball_y.copy()
    fillable = np.isnan(ball_x) & ~np.isnan(clean.ball_x)
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
                    if rng.random() < fill.inlier_frac:
                        dx = rng.normal(0, fill.inlier_sigma, len(seg))
                        dy = rng.normal(0, fill.inlier_sigma, len(seg))
                    else:
                        r = fill.outlier_min + np.exp(rng.normal(np.log(fill.outlier_median), fill.outlier_sigma))
                        angle = rng.uniform(0, 2 * np.pi)
                        r = min(r, 40.0)
                        dx = r * np.cos(angle) + rng.normal(0, fill.jitter, len(seg))
                        dy = r * np.sin(angle) + rng.normal(0, fill.jitter, len(seg))
                    ball_x[seg] = clean.ball_x[seg] + dx
                    ball_y[seg] = clean.ball_y[seg] + dy
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
