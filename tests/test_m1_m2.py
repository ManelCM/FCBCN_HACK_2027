from dataclasses import replace

import numpy as np

from hackbarca.io.degrade import DegradeParams, _ball_signal
from hackbarca.io.load_match import AWAY, HOME, MatchTracking
from hackbarca.pipeline.atd_stats import ball_dynamics


def test_corr_smooth_removes_apparent_speed_of_still_ball():
    n = 3000
    x, y = np.full(n, 50.0), np.full(n, 30.0)
    base = DegradeParams(ball_noise=0.0, ball_corr_noise=0.8, ball_corr_tau=75.0)
    rough = _ball_signal(np.random.default_rng(0), x, y, base)
    smooth = _ball_signal(np.random.default_rng(0), x, y, replace(base, ball_corr_smooth=10.0))
    step = lambda b: np.median(np.hypot(np.diff(b[0]), np.diff(b[1])))   # noqa: E731
    assert step(smooth) < step(rough) / 3
    # sin suavizar no cambia nada respecto a la version anterior (mismos numeros aleatorios)
    again = _ball_signal(np.random.default_rng(0), x, y, base)
    assert np.array_equal(rough[0], again[0])


def test_ball_dynamics_slow_runs():
    fps = 25
    n = 60 * fps
    x = np.cumsum(np.full(n, 0.3))                  # 7,5 m/s
    x[600:700] = x[599] + np.arange(100) * 0.01     # 4 s de balon lento (0,25 m/s)
    x[700:] = x[699] + np.cumsum(np.full(n - 700, 0.3))
    y = np.full(n, 30.0)
    graded = np.ones(n, bool)
    d = ball_dynamics(x, y, graded, live=graded, fps=fps)
    assert abs(d["live_slow_runs_per_min"] - 1.0) < 0.01   # el primer frame no tiene paso
    assert 0.05 < d["live_slow_share"] < 0.08
    assert d["speed_p50"] > 7


def _match(n, home_xy, away_xy):
    px = np.column_stack([home_xy[0], away_xy[0]])
    py = np.column_stack([home_xy[1], away_xy[1]])
    return MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                         team_of_track=np.array([HOME, AWAY], dtype=np.int8), players_x=px, players_y=py,
                         players_speed=np.zeros_like(px), ball_x=np.zeros(n), ball_y=np.zeros(n),
                         ball_speed=np.zeros(n), period=np.ones(n, dtype=np.int8))


def test_control_features_see_a_steal():
    from hackbarca.features.control import control_features
    n = 300
    t = np.arange(n)
    bx = 50 + t * 0.04                               # balon conducido a 1 m/s
    by = np.full(n, 30.0)
    home = (bx - 0.5, by.copy())                     # el local lo lleva pegado...
    away = (bx + 5.0, by.copy())
    home[0][150:] = bx[150:] - 6.0                    # ...hasta que el visitante se lo quita en t=150
    away[0][150:] = bx[150:] + 0.6
    m = _match(n, home, away)
    f = control_features(m, bx, by, np.ones(n, bool))
    assert f["ctrl_h"][2:150].all() and not f["ctrl_h"][155:].any()   # sin velocidad en los bordes
    assert f["ctrl_a"][155:-2].all()
    assert f["ctrl_chg25_a"][150] > 0.9 and f["ctrl_chg25_h"][150] < -0.9
    assert round(np.exp(f["ctrl_run_h"][10]) - 1) == 148   # racha de los frames 2-149


def test_skillcorner_resample_10_to_25_fps():
    from hackbarca.io.skillcorner import _resample
    t = np.arange(0, 2.0, 0.1)
    v = np.column_stack([t * 10, t * 0 + 3.0])
    v[5, 0] = np.nan                                  # un hueco de un frame
    grid = np.arange(0, 1.9, 0.04)
    out = _resample(t, v, grid)
    ok = np.isfinite(out[:, 0])
    assert np.allclose(out[ok, 0], grid[ok] * 10)
    assert not ok[(grid > 0.4) & (grid < 0.6)].any()    # junto al hueco no se inventa
    assert np.allclose(out[:, 1], 3.0)
