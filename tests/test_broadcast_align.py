import numpy as np

from hackbarca.io.broadcast_align import align, alignment_report, fuse_ball, shot_segments
from hackbarca.io.broadcast_sim import simulate_broadcast
from hackbarca.io.load_match import MatchTracking


def _smooth_match(n=6000, seed=0):
    """22 jugadores con trayectorias suaves (paseo aleatorio con inercia) y un balon."""
    rng = np.random.default_rng(seed)
    pos = np.column_stack([rng.uniform(10, 95, 22), rng.uniform(8, 60, 22)])
    vel = np.zeros_like(pos)
    px, py = np.empty((n, 22)), np.empty((n, 22))
    for f in range(n):
        vel = 0.95 * vel + rng.normal(0, 0.03, vel.shape)
        pos = np.clip(pos + vel, [1, 1], [104, 67])
        px[f], py[f] = pos[:, 0], pos[:, 1]
    owner = (np.arange(n) // 150) % 22
    true_x = px[np.arange(n), owner] + 0.5
    true_y = py[np.arange(n), owner]
    ball_x, ball_y = true_x.copy(), true_y.copy()
    ball_x[rng.random(n) < 0.4] = np.nan
    ball_y[np.isnan(ball_x)] = np.nan
    team = np.repeat(np.array([0, 1], dtype=np.int8), 11)
    match = MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                          team_of_track=team, players_x=px, players_y=py,
                          players_speed=np.full_like(px, np.nan), ball_x=ball_x, ball_y=ball_y,
                          ball_speed=np.full(n, np.nan))
    match.true_ball = (true_x, true_y)
    return match


def test_shot_segments_split_at_cuts():
    t = _smooth_match(n=3000)
    b, truth = simulate_broadcast(t, seed=1)
    segments = shot_segments(b)
    assert len(segments) >= 3
    for start, stop in segments:   # ningun plano mezcla frames de dos planos verdaderos
        live = truth[start:stop]
        live = live[live >= 0]
        if len(live) > 1:
            assert np.all(np.diff(live) == 1)


def test_align_recovers_live_shots_and_rejects_the_rest():
    t = _smooth_match()
    b, truth = simulate_broadcast(t, seed=2)
    result = align(t, b)
    mapped = np.flatnonzero(result.tactical_to_broadcast >= 0)
    assert len(mapped) > 0
    # cada frame tactico emparejado apunta a su frame broadcast con error <= 2 frames (0,08 s)
    true_t = truth[result.tactical_to_broadcast[mapped]]
    correct = (true_t >= 0) & (np.abs(true_t - mapped) <= 2)
    assert correct.mean() > 0.95
    # y se recupera la mayoria de los frames en directo
    live = np.unique(truth[truth >= 0])
    assert len(mapped) / len(live) > 0.8


def test_fuse_ball_fills_only_missing_frames():
    t = _smooth_match()
    b, _ = simulate_broadcast(t, seed=3, true_ball=t.true_ball)
    result = align(t, b)
    fused, source = fuse_ball(t, b, result)
    had = ~np.isnan(t.ball_x)
    np.testing.assert_array_equal(fused.ball_x[had], t.ball_x[had])
    assert (source == 1).sum() > 0
    assert np.isnan(t.ball_x[source == 1]).all()
    report = alignment_report(t, b, result)
    assert report["frac_huecos_rellenables"] > 0.2
    assert report["dist_balon_mediana_m"] < 1.5
