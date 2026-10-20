import numpy as np
import pandas as pd

from hackbarca.io.degrade import DegradeParams, degrade
from hackbarca.io.external import GT_A, GT_D, GT_H, GT_X, gt_from_events
from hackbarca.io.load_match import AWAY, HOME, MatchTracking
from hackbarca.models.baseline import NOBODY, nearest_team, predict_baseline


def _events(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])


def test_gt_from_events_dead_ball_and_owner():
    events = _events([
        (HOME, "SET PIECE", "KICK OFF", 1, 10, 10),
        (HOME, "PASS", "", 1, 10, 20),
        (AWAY, "CHALLENGE", "AERIAL-WON", 1, 25, 25),   # duelo: no cambia la posesion
        (AWAY, "RECOVERY", "INTERCEPTION", 1, 30, 30),
        (AWAY, "BALL OUT", "", 1, 40, 50),
        (HOME, "SET PIECE", "THROW IN", 1, 70, 70),
        (HOME, "PASS", "", 1, 70, 80),
    ])
    gt = gt_from_events(events, 100)
    assert (gt[:10] == GT_X).all() and (gt[81:] == GT_X).all()
    assert (gt[10:30] == GT_H).all()
    assert (gt[30:50] == GT_A).all()
    assert (gt[50:70] == GT_D).all()
    assert (gt[70:81] == GT_H).all()


def test_gt_from_events_foul_starts_dead_ball_at_the_foul():
    events = _events([
        (HOME, "SET PIECE", "KICK OFF", 1, 0, 0),
        (HOME, "PASS", "", 1, 0, 10),
        (AWAY, "FAULT RECEIVED", "", 1, 15, 15),
        (HOME, "CARD", "YELLOW", 1, 30, 30),
        (AWAY, "SET PIECE", "FREE KICK", 1, 40, 40),
        (AWAY, "PASS", "", 1, 40, 50),
    ])
    gt = gt_from_events(events, 60)
    assert (gt[15:40] == GT_D).all()
    assert gt[40] == GT_A


def _synthetic_match(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    px = rng.uniform(0, 105, (n, 22))
    py = rng.uniform(0, 68, (n, 22))
    team = np.repeat(np.array([HOME, AWAY], dtype=np.int8), 11)
    ball_x, ball_y = px[:, 0] + 0.5, py[:, 0]
    ball_x[100:200] = np.nan
    ball_y[100:200] = np.nan
    return MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                         team_of_track=team, players_x=px, players_y=py,
                         players_speed=np.full_like(px, np.nan), ball_x=ball_x, ball_y=ball_y,
                         ball_speed=np.full(n, np.nan))


def test_nearest_team_and_baseline():
    match = _synthetic_match()
    near = nearest_team(match)
    assert (near[100:200] == NOBODY).all()
    assert np.mean(near[:100] == HOME) > 0.95
    pred = predict_baseline(match, window=0)
    assert set(np.unique(pred)) <= {0, 1, 2}
    assert (pred[100:200] == 2).all()


def test_degrade_is_deterministic_and_anonymous():
    match = _synthetic_match()
    params = DegradeParams(ball_drop_rate=0.3, tracklet_mean=200)
    a, b = degrade(match, params, seed=1), degrade(match, params, seed=1)
    np.testing.assert_array_equal(np.isnan(a.ball_x), np.isnan(b.ball_x))
    np.testing.assert_array_equal(a.players_x, b.players_x)
    assert a.n_slots == 2 * params.slots_per_team
    assert np.isnan(a.ball_x).mean() > np.isnan(match.ball_x).mean()
    # cada slot solo pertenece a un equipo y nunca tiene dos tracklets a la vez (una columna)
    assert set(np.unique(a.team_of_track)) == {HOME, AWAY}
