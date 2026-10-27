import numpy as np

from hackbarca.io.ball_action_signal import BallActions, ball_action_features, pick_events, simulate_ball_actions


def _gt():
    # 2 min: 40 s local, 10 s parada, 40 s visitante, 30 s local
    return np.concatenate([np.zeros(1000), np.full(250, 2), np.ones(1000), np.zeros(750)]).astype(int)


def test_simulation_rate_and_accuracy():
    gt = np.tile(_gt(), 20)
    team = simulate_ball_actions(gt, BallActions(err_near=0, err_mid=0, err_far=0.1), seed=1)
    ev = np.flatnonzero(np.isfinite(team))
    live = np.isin(gt, (0, 1))
    rate = np.isin(gt[ev], (0, 1)).sum() / (live.sum() / 25 / 60)
    assert 11 < rate < 15          # 14,5/min menos lo que quita la separacion de 1 s
    ok = live[ev]
    assert abs((team[ev][ok] == gt[ev][ok]).mean() - 0.9) < 0.03
    again = simulate_ball_actions(gt, BallActions(err_near=0, err_mid=0, err_far=0.1), seed=1)
    np.testing.assert_array_equal(np.isnan(team), np.isnan(again))


def test_features_last_next():
    team = np.full(200, np.nan)
    team[50], team[150] = 0, 1
    f = ball_action_features(team)
    assert f["ba_last_h"][100] == 1 and f["ba_next_h"][100] == 0 and f["ba_flip"][100] == 1
    assert np.isnan(f["ba_last_h"][10]) and abs(f["ba_next_s"][10] - 40 / 25) < 1e-6


def test_pick_events():
    fr = np.arange(0, 1000, 2)
    p = np.zeros((len(fr), 25))
    p[:, 0] = 1
    p[100, 2], p[101, 2], p[300, 13] = 0.6, 0.5, 0.4
    keep, cls, side = pick_events(fr, p, 0.3)
    assert list(fr[keep]) == [200, 600] and list(cls) == [0, 6] and list(side) == [1, 0]
