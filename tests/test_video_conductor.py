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
