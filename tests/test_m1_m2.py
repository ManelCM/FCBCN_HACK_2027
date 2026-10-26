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


def test_real_signal_gates_on_synthetic_prediction():
    from hackbarca.pipeline import real_signals as rs
    n = 200000
    pred = np.full(n, 3, dtype=np.int8)
    for a, b in rs.GRADED:
        pred[a:b] = 0
    pred[20000:21000] = 2                               # D largo
    pred[30000:30050] = 2                               # D corto entre vivos
    pmax = np.full(n, 0.95)
    known = np.ones(n, bool)
    public = np.full(n, 3, dtype=np.int8)
    ctx = rs.contexts(pred, pmax, known, public)
    s = np.where(pred == 2, 1.0, 0.0)
    row = rs.gate_dead("x", s, ctx)
    assert row["auc_seguro"] == 1.0 and row["D_cortos_entre_vivo0_y_paron1"] == 1.0
    team = np.where(pred == 0, 1.0, np.nan)
    assert rs.gate_team("t", team, ctx, pred, public)["acuerdo_seguro"] == 1.0


def test_broadcast_signal_follows_state():
    from hackbarca.io.broadcast_signal import broadcast_features, shape, simulate_broadcast_count
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(80):                               # 80 ciclos de juego (40 s) y paron (20 s)
        parts += [np.full(1000, rng.integers(0, 2)), np.full(500, 2)]
    gt = np.concatenate(parts).astype(np.int8)
    c = simulate_broadcast_count(gt, seed=1)
    s = shape(c, gt)
    assert s["vivo"]["ge10"] > 0.8 and s["D"]["ge10"] < 0.45 and s["D"]["le3"] > 0.5
    assert np.array_equal(c, simulate_broadcast_count(gt, seed=1))          # determinista
    f = broadcast_features(c)
    assert {"bc_n", "bc_wide_m75", "bc_wide_chg75"} <= set(f.columns) and len(f) == len(gt)


def test_transition_sees_the_winner_go_forward():
    from hackbarca.features.coach import extra_features, transition_features
    n = 400
    t = np.arange(n)
    # 10 jugadores por equipo; el local ataca hacia x creciente. Hasta t=200 el visitante avanza
    # (ataca hacia x decreciente) y el local repliega; luego roba el local y avanza
    hx = np.where(t < 200, 50 - 0.04 * t, 42 + 0.08 * (t - 200))[:, None] + np.linspace(-20, 5, 10)[None, :]
    ax = hx + 10
    hy = np.tile(np.linspace(5, 63, 10), (n, 1))
    m = _match(n, (hx[:, 0], hy[:, 0]), (ax[:, 0], hy[:, 0]))
    from dataclasses import replace
    m = replace(m, team_of_track=np.array([0] * 10 + [1] * 10, dtype=np.int8),
                players_x=np.concatenate([hx, ax], 1), players_y=np.concatenate([hy, hy], 1),
                players_speed=np.zeros((n, 20)), period=np.ones(n, dtype=np.int8))
    signs = {"h": np.ones(n), "a": -np.ones(n)}
    f = transition_features(m, np.full(n, np.nan), np.full(n, np.nan), signs)
    assert f["tr_cen_v_chg75_h"][200] > 2.0           # el local pasa de replegar a salir
    cols = extra_features(m, ("transition", "gamestate", "territory"))
    assert len(cols) == n and "gs_speed_p0.8" in cols and not any("tilt" in c for c in cols)
