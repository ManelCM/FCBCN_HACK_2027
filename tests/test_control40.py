import numpy as np

from hackbarca.features.control40 import control40_features
from hackbarca.io.load_match import AWAY, HOME, MatchTracking


def _match(n, home_xy, away_xy, period=None):
    px = np.column_stack([home_xy[0], away_xy[0]])
    py = np.column_stack([home_xy[1], away_xy[1]])
    return MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                         team_of_track=np.array([HOME, AWAY], dtype=np.int8), players_x=px, players_y=py,
                         players_speed=np.zeros_like(px), ball_x=np.zeros(n), ball_y=np.zeros(n),
                         ball_speed=np.zeros(n), period=period)


def _steal(n=300):
    t = np.arange(n)
    bx = 50 + t * 0.04                               # balon conducido a 1 m/s
    by = np.full(n, 30.0)
    home = (bx - 0.5, by.copy())                     # el local lo lleva pegado...
    away = (bx + 5.0, by.copy())
    home[0][150:] = bx[150:] - 6.0                    # ...hasta que el visitante se lo quita en t=150
    away[0][150:] = bx[150:] + 0.6
    return bx, by, home, away


def test_forty_columns_in_pairs():
    bx, by, home, away = _steal()
    f = control40_features(_match(len(bx), home, away), bx, by)
    assert len(f) == 40
    assert sum(k.startswith("ctl_") for k in f) == 32 and sum(k.startswith("response_") for k in f) == 8
    assert all(k.endswith(("_h", "_a")) for k in f)
    assert {k[:-2] for k in f if k.endswith("_h")} == {k[:-2] for k in f if k.endswith("_a")}


def test_steal_is_seen():
    bx, by, home, away = _steal()
    f = control40_features(_match(len(bx), home, away), bx, by)
    assert f["ctl_close_h"][:150].all() and not f["ctl_close_h"][150:].any()
    assert f["ctl_close_a"][150:].all()
    assert f["ctl_close_fraction3_h"][100] == 1.0 and f["ctl_close_fraction3_a"][250] == 1.0
    assert f["ctl_sup2_f3_a"][149] > f["ctl_sup2_b3_a"][149]


def test_swapping_teams_swaps_columns():
    bx, by, home, away = _steal()
    f = control40_features(_match(len(bx), home, away), bx, by)
    g = control40_features(_match(len(bx), away, home), bx, by)
    for k in f:
        other = k[:-2] + ("_a" if k.endswith("_h") else "_h")
        np.testing.assert_allclose(f[k], g[other], equal_nan=True)


def test_hidden_ball_and_periods():
    bx, by, home, away = _steal()
    bx, by = bx.copy(), by.copy()
    bx[200:230], by[200:230] = np.nan, np.nan
    period = np.where(np.arange(len(bx)) < 150, 1, 2).astype(np.int8)
    f = control40_features(_match(len(bx), home, away, period), bx, by)
    assert np.isnan(f["ctl_dist_a"][210]) and np.isnan(f["ctl_close_a"][210])
    # la ventana pasada del primer frame de la 2.a parte no ve la 1.a
    assert np.isnan(f["ctl_sup2_b1_a"][150])
    assert np.isnan(f["response_dspeed1_h"][150])
