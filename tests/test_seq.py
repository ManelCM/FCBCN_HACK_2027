import numpy as np
import pytest

torch = pytest.importorskip("torch")  # grupo opcional 'vision'


def _match(n=1500, seed=0):
    from hackbarca.io.load_match import MatchTracking
    rng = np.random.default_rng(seed)
    px = rng.uniform(0, 105, (n, 22)).cumsum(0) % 105
    py = rng.uniform(0, 68, (n, 22)).cumsum(0) % 68
    team = np.repeat(np.array([0, 1], dtype=np.int8), 11)
    bx, by = px[:, 0] + 0.5, py[:, 0].copy()
    bx[300:400] = np.nan
    by[300:400] = np.nan
    return MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0, team_of_track=team,
                         players_x=px, players_y=py, players_speed=np.full_like(px, np.nan),
                         ball_x=bx, ball_y=by, ball_speed=np.full(n, np.nan))


def test_seq_model_trains_and_predicts_deterministically():
    from hackbarca.models.seq import predict_seq, seq_inputs, train_seq
    m = _match()
    gt = np.where(np.arange(m.n_frames) % 500 < 250, 0, 1).astype(np.int8)
    tokens, mask, glob, y, ball = seq_inputs(m, gt, m)
    model = train_seq([(tokens, mask, glob, y, ball)], epochs=1, device="cpu")
    p1 = predict_seq(model, tokens, mask, glob, m.n_frames)
    p2 = predict_seq(model, tokens, mask, glob, m.n_frames)
    assert p1.shape == (m.n_frames, 3) and np.allclose(p1.sum(1), 1)
    np.testing.assert_array_equal(p1, p2)
