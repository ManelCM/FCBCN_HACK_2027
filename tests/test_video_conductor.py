import numpy as np

from hackbarca.features.frame_features import swap_teams
from hackbarca.io.video_conductor import _runs, conductor_features


def test_runs_keep_share():
    rng = np.random.default_rng(0)
    for share, mean in ((0.0, 3), (0.03, 3), (0.32, 10), (0.94, 7.5), (1.0, 3)):
        m = _runs(rng, 200_000, share, mean)
        assert abs(m.mean() - share) < 0.01


def test_features_follow_team_swap():
    rng = np.random.default_rng(1)
    team = np.full(5000, np.nan)
    team[::5] = rng.integers(0, 2, 1000)
    team[rng.random(5000) < 0.5] = np.nan
    f = conductor_features(team)
    g = conductor_features(np.where(np.isfinite(team), 1 - team, np.nan))
    pd_equal = np.allclose(swap_teams(f).to_numpy(), g.to_numpy(), equal_nan=True)
    assert pd_equal
    assert np.nanmax(f["vc_m50_h"] + f["vc_m50_a"]) <= 1.0 + 1e-6


def test_stress_errors_near_changes():
    from types import SimpleNamespace

    from hackbarca.io.video_conductor import VideoConductor, simulate_conductor
    n = 25 * 600
    gt = np.repeat(np.arange(n // 250) % 2, 250)  # cambio H<->A cada 10 s
    match = SimpleNamespace(n_frames=n, players_x=np.zeros((n, 2)), players_y=np.array([[0.0, 30.0]] * n),
                            team_of_track=np.array([0, 1]), ball_x=np.zeros(n), ball_y=np.zeros(n))
    match.players_y[gt == 1] = [30.0, 0.0]
    clean = VideoConductor(keep=1.0, err=0.0)
    base = simulate_conductor(match, gt, clean)
    ok = np.isfinite(base)
    assert np.all(base[ok] == gt[ok])
    stress = simulate_conductor(match, gt, VideoConductor(keep=1.0, err=0.0, change_err=0.5))
    wrong = np.isfinite(stress) & (stress != gt)
    near = np.zeros(n, dtype=bool)
    for c in range(250, n, 250):
        near[max(c - 37, 0):c + 38] = True
    seen = np.isfinite(stress)
    assert wrong[near & seen].mean() > 0.3 and wrong[~near].sum() == 0
