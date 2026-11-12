import numpy as np

from hackbarca.io.load_match import MatchTracking
from hackbarca.io.video_fill import VideoFill, apply_video_fill, simulate_video_fill


def _match(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    bx, by = 52.5 + 30 * np.sin(t / 400), 34 + 20 * np.cos(t / 300)
    px = rng.uniform(0, 105, (n, 4))
    return MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                         team_of_track=np.array([0, 0, 1, 1], np.int8), players_x=px, players_y=px.copy(),
                         players_speed=np.full((n, 4), np.nan), ball_x=bx, ball_y=by, ball_speed=np.full(n, np.nan))


def test_simulated_fill_matches_measured_shape():
    from dataclasses import replace
    clean = _match()
    gaps = np.zeros(clean.n_frames, bool)
    for a in range(0, clean.n_frames, 1000):
        gaps[a + 200:a + 700] = True     # huecos de 500 frames
    degraded = replace(clean, ball_x=np.where(gaps, np.nan, clean.ball_x), ball_y=np.where(gaps, np.nan, clean.ball_y))
    filled = simulate_video_fill(degraded, clean, VideoFill(), seed=3)
    np.testing.assert_array_equal(filled.ball_x[~gaps], clean.ball_x[~gaps])   # fuera de huecos no toca
    covered = gaps & ~np.isnan(filled.ball_x)
    assert 0.45 < covered.sum() / gaps.sum() < 0.65                           # ~56 % medido en el reto
    err = np.hypot(filled.ball_x[covered] - clean.ball_x[covered], filled.ball_y[covered] - clean.ball_y[covered])
    assert 0.4 < np.median(err) < 1.5
    assert 0.55 < np.mean(err < 2) < 0.9
    again = simulate_video_fill(degraded, clean, VideoFill(), seed=3)
    np.testing.assert_array_equal(np.isnan(again.ball_x), np.isnan(filled.ball_x))   # determinista


def test_simulated_fill_is_smooth_and_holds_dead_ball():
    from dataclasses import replace
    clean = _match(6000)
    gaps = np.zeros(clean.n_frames, bool)
    gaps[1000:5000] = True
    still = clean.ball_x * 0 + 50.0                                            # balon parado: velocidad 0
    clean = replace(clean, ball_x=still.copy(), ball_y=still.copy())
    cx = clean.ball_x.copy()
    cx[2000:3000] = np.nan                                                     # paron en el limpio
    clean = replace(clean, ball_x=cx, ball_y=np.where(np.isnan(cx), np.nan, clean.ball_y))
    degraded = replace(clean, ball_x=np.where(gaps, np.nan, clean.ball_x), ball_y=np.where(gaps, np.nan, clean.ball_y))
    filled = simulate_video_fill(degraded, clean, VideoFill(inlier_frac=1.0), seed=1)
    x, y = filled.ball_x, filled.ball_y
    both = ~np.isnan(x[1:]) & ~np.isnan(x[:-1]) & gaps[1:]
    speed = np.hypot(np.diff(x), np.diff(y))[both] * 25
    assert np.median(speed) < 1.5                                              # sin saltos de ruido por frame
    assert (~np.isnan(x[2000:3000])).mean() > 0.3                              # el paron tambien se rellena


def test_apply_video_fill_only_touches_atd_gaps(tmp_path):
    from dataclasses import replace
    clean = _match(100)
    bx = clean.ball_x.copy()
    bx[10:20] = np.nan
    match = replace(clean, ball_x=bx, ball_y=np.where(np.isnan(bx), np.nan, clean.ball_y))
    path = str(tmp_path / "video_ball.npz")
    frames = np.arange(5, 25)
    x = np.full(len(frames), 1.0)
    x[frames == 15] = np.nan                                                   # el detector tampoco lo ve
    np.savez(path, frame=frames, x=x, y=np.full(len(frames), 2.0))
    out = apply_video_fill(match, path)
    np.testing.assert_array_equal(out.ball_x[:10], match.ball_x[:10])          # el ATD manda donde ve el balon
    assert np.isnan(out.ball_x[15])
    assert (out.ball_x[[10, 14, 16, 19]] == 1.0).all()
    np.testing.assert_array_equal(out.ball_x[20:], match.ball_x[20:])


def test_v3_skips_fast_live_ball_and_carries_dead_ball():
    from dataclasses import replace
    from hackbarca.io.video_fill import VIDEO_FILL
    n = 3000
    t = np.arange(n)
    bx = np.where(t < 1500, 10 + 0.6 * t, np.nan)              # 15 m/s en juego vivo; despues paron
    bx[2900:] = 60.0                                           # reanudacion en x=60
    by = np.where(np.isnan(bx), np.nan, 30.0)
    px = np.column_stack([np.full(n, 61.0), np.full(n, 20.0)])  # alguien junto al punto de reanudacion
    py = np.column_stack([np.full(n, 30.0), np.full(n, 50.0)])
    clean = MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                          team_of_track=np.array([0, 1], np.int8), players_x=px, players_y=py,
                          players_speed=np.full((n, 2), np.nan), ball_x=bx, ball_y=by, ball_speed=np.full(n, np.nan))
    gaps = (t >= 500) & (t < 2900)
    degraded = replace(clean, ball_x=np.where(gaps, np.nan, bx), ball_y=np.where(gaps, np.nan, by))
    filled = simulate_video_fill(degraded, clean, VIDEO_FILL, seed=0)
    assert np.isnan(filled.ball_x[500:1500]).all()             # balon rapido en juego vivo: el video no lo ve
    dead = ~np.isnan(filled.ball_x[1500:2900])
    assert dead.mean() > 0.3                                   # el paron si se rellena
    d = np.hypot(filled.ball_x[1500:2900][dead] - 61.0, filled.ball_y[1500:2900][dead] - 30.0)
    assert np.median(d) < 3.0                                  # quieto en la reanudacion o con quien va a sacar
