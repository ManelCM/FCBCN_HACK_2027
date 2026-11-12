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


def test_long_ball_model_makes_few_long_gaps():
    match = _synthetic_match(n=20000)
    params = DegradeParams(ball_model="long", gap_median=200, short_gap_frac=0.0,
                           ball_drop_rate=0.4, dead_ball_visible=1.0)
    a, b = degrade(match, params, seed=3), degrade(match, params, seed=3)
    missing = np.isnan(a.ball_x)
    np.testing.assert_array_equal(missing, np.isnan(b.ball_x))
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert np.median(lengths) > 80
    assert 0.2 < missing.mean() < 0.6


def test_swap_teams_is_an_involution():
    from hackbarca.features.frame_features import frame_features, swap_teams
    df = frame_features(_synthetic_match(n=600), windows=[(100, 225)])
    back = swap_teams(swap_teams(df))
    np.testing.assert_allclose(back.to_numpy(), df.to_numpy(), equal_nan=True)
    swapped = swap_teams(df)
    np.testing.assert_allclose(swapped["d1_h"], df["d1_a"], equal_nan=True)
    np.testing.assert_allclose(swapped["near_sign"], -df["near_sign"], equal_nan=True)


def test_v2_touch_detected_at_a_kick():
    from hackbarca.features.frame_features import frame_features
    match = _synthetic_match(n=400)
    # balon rodando a 2 m/s hacia +x y patada a 20 m/s hacia +y en el frame 200
    t = np.arange(400)
    bx = np.where(t < 200, 30 + 2 * t / 25, 30 + 2 * 200 / 25)
    by = np.where(t < 200, 30.0, 30 + 20 * (t - 200) / 25)
    match = MatchTracking(n_frames=400, fps=25, pitch_length=105.0, pitch_width=68.0,
                          team_of_track=match.team_of_track, players_x=match.players_x[:400],
                          players_y=match.players_y[:400], players_speed=match.players_speed[:400],
                          ball_x=bx.astype(float), ball_y=np.minimum(by, 67.0),
                          ball_speed=np.full(400, np.nan))
    df = frame_features(match, version=2)
    peak = int(np.nanargmax(df["touch_dv"].to_numpy()))
    assert abs(peak - 200) <= 2
    assert df["since_touch_log"].iloc[210] < df["since_touch_log"].iloc[190]


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


def test_controlled_possession_absorbs_uncontrolled_steals():
    from hackbarca.io.external import controlled_possession
    H, A, D = GT_H, GT_A, GT_D
    gt = np.array([H] * 10 + [A] * 2 + [H] * 5 + [D] * 3 + [A] * 2 + [H] * 8 + [A] * 6, dtype=np.int8)
    out = controlled_possession(gt, 4)
    # robo de 2 frames sin control -> sigue H; tras D el poseedor vale aunque dure poco
    assert (out[:17] == H).all()
    assert (out[17:20] == D).all()
    assert (out[20:22] == A).all()
    assert (out[22:30] == H).all() and (out[30:] == A).all()
    assert np.array_equal(controlled_possession(gt, 0), gt)


def _fake_dfl_dataset():
    """TrackingDataset minimo con la forma que da kloppy para DFL: dos partes, flags por frame."""
    from types import SimpleNamespace as NS
    home, away = NS(players=[NS(player_id="h1")]), NS(players=[NS(player_id="a1")])
    p1, p2 = NS(id=1), NS(id=2)
    rows = [  # (parte, estado, duenio)
        (p1, "BallState.ALIVE", home), (p1, "BallState.ALIVE", away), (p1, "BallState.DEAD", away),
        (p2, "BallState.ALIVE", away), (p2, "BallState.ALIVE", home),
    ]
    records = [NS(period=p, ball_state=s, ball_owning_team=o) for p, s, o in rows]
    df = pd.DataFrame({"ball_x": [0.5] * 5, "ball_y": [0.5] * 5, "h1_x": [0.1] * 5, "h1_y": [0.2] * 5,
                       "a1_x": [0.9] * 5, "a1_y": [0.8] * 5})
    meta = NS(teams=[home, away], frame_rate=25.0)
    return NS(records=records, metadata=meta, to_df=lambda engine=None: df)


def test_dfl_to_bench_gt_from_tracking_flags_and_halves():
    from hackbarca.io.external import dfl_to_bench
    match, gt = dfl_to_bench(_fake_dfl_dataset())
    # una fila X entre partes para que possession_metrics no las una
    assert gt.tolist() == [GT_H, GT_A, GT_D, GT_X, GT_A, GT_H]
    assert match.n_frames == 6 and match.period.tolist() == [1, 1, 1, 0, 2, 2]
    assert np.isnan(match.ball_x[2]) and np.isnan(match.ball_x[3])  # sin balon en D (como Metrica)
    assert np.isclose(match.ball_x[0], 52.5) and np.isclose(match.players_x[0, 0], 10.5)
    assert match.team_of_track.tolist() == [HOME, AWAY]


def test_bench_v3_names_dispatch():
    from hackbarca.io.external import is_dfl
    from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
    assert len(BENCH_GAMES_V3) == 10
    assert [is_dfl(g) for g in BENCH_GAMES_V3] == [False] * 3 + [True] * 7


def test_v5_change_features_see_a_turnover():
    from hackbarca.features.frame_features import frame_features, swap_teams
    n = 600
    t = np.arange(n)
    team = np.repeat(np.array([HOME, AWAY], dtype=np.int8), 11)
    px = np.empty((n, 22))
    px[:, :11] = 30.0 + np.where(t >= 300, (t - 300) * 0.05, 0.0)[:, None]  # el local se adelanta
    px[:, 11:] = 70.0
    py = np.tile(np.linspace(5, 63, 11), 2)[None, :].repeat(n, 0)
    ball_x = np.where(t < 300, 40 + 0.08 * t, 64 - 0.08 * (t - 300))       # el balon da la vuelta
    m = MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0, team_of_track=team,
                      players_x=px, players_y=py, players_speed=np.full_like(px, np.nan),
                      ball_x=ball_x, ball_y=np.full(n, 34.0), ball_speed=np.full(n, np.nan))
    df = frame_features(m, version="touch+ball+blocks")
    assert df["ball_prog_chg25"].iloc[300] < -3     # el local atacaba (+x) y el balon va hacia atras
    assert df["blk_back_chg75_h"].iloc[300] > 1 and abs(df["blk_back_chg75_a"].iloc[300]) < 1e-6
    swapped = swap_teams(df)
    np.testing.assert_allclose(swapped["ball_prog_chg25"], -df["ball_prog_chg25"], equal_nan=True)
    np.testing.assert_allclose(swap_teams(swapped).to_numpy(), df.to_numpy(), equal_nan=True)


def test_layout_features_see_a_static_corner():
    from hackbarca.features.frame_features import frame_features, swap_teams
    n = 300
    team = np.repeat(np.array([HOME, AWAY], dtype=np.int8), 11)
    rng = np.random.default_rng(3)
    px = np.empty((n, 22))
    py = np.empty((n, 22))
    px[:, :] = rng.uniform(90, 104, 22)[None, :]          # todos en el area derecha, quietos
    py[:, :] = rng.uniform(20, 48, 22)[None, :]
    px[:, 0], py[:, 0] = 105.0, 0.3                        # el que saca el corner, en la banda
    m = MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0, team_of_track=team,
                      players_x=px, players_y=py, players_speed=np.full_like(px, np.nan),
                      ball_x=np.full(n, np.nan), ball_y=np.full(n, np.nan), ball_speed=np.full(n, np.nan))
    df = frame_features(m, version="touch+restart")
    mid = n // 2
    assert df["frac_still"].iloc[mid] == 1.0
    assert df["box_r"].iloc[mid] >= 15 and df["box_l"].iloc[mid] == 0
    assert df["touch_min"].iloc[mid] < 0.5
    np.testing.assert_allclose(swap_teams(swap_teams(df)).to_numpy(), df.to_numpy(), equal_nan=True)


def test_feature_cache_roundtrip_and_key(tmp_path):
    from hackbarca.features.frame_features import frame_features
    from hackbarca.pipeline.bench_config import BENCH_DEGRADE
    from hackbarca.pipeline.tune_l3 import feature_path, read_features
    df = frame_features(_synthetic_match(n=400), version=3)
    path = str(tmp_path / "g.npz")
    np.savez(path, columns=np.array(df.columns, dtype=str), values=df.to_numpy(np.float32))
    back = read_features(path)
    assert list(back.columns) == list(df.columns)
    np.testing.assert_array_equal(back.to_numpy(), df.to_numpy(np.float32))
    a = feature_path("1", 0, 3, False, BENCH_DEGRADE)
    assert a == feature_path("1", 0, 3, False, BENCH_DEGRADE)
    assert a != feature_path("1", 0, 5, False, BENCH_DEGRADE) != feature_path("1", 1, 3, False, BENCH_DEGRADE)


def test_v8_tactical_features_goalkeeper_line_and_sprints():
    from hackbarca.features.frame_features import frame_features, swap_teams
    n = 400
    t = np.arange(n)
    team = np.repeat(np.array([HOME, AWAY], dtype=np.int8), 11)
    px = np.empty((n, 22))
    py = np.tile(np.linspace(5, 63, 11), 2)[None, :].repeat(n, 0)
    px[:, 0] = np.where(t < 200, 20.0, np.maximum(20.0 - (t - 200) * 0.3, 1.0))  # portero local vuelve a su porteria
    px[:, 1:5] = 30.0 + np.where(t >= 200, (t - 200) * 0.25, 0.0)[:, None]      # su linea adelanta a 6 m/s
    px[:, 5:11] = 45.0
    px[:, 11:] = 70.0
    px[:, 21] = 100.0                                                            # portero visitante quieto
    m = MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0, team_of_track=team,
                      players_x=px, players_y=py, players_speed=np.full_like(px, np.nan),
                      ball_x=np.full(n, 50.0), ball_y=np.full(n, 34.0), ball_speed=np.full(n, np.nan))
    df = frame_features(m, version="touch+ball+players+blocks+structure+restart")
    assert df["gk_goal_chg75_h"].iloc[200] < -3          # el portero local se acerca a su linea de gol
    assert df["blk_back_chg75_h"].iloc[200] > 3 and abs(df["blk_back_chg75_a"].iloc[200]) < 1e-6
    assert df["sprint_n_h"].iloc[300] >= 4 and df["sprint_n_a"].iloc[300] == 0
    assert df["ball_still_m25"].iloc[200] == 1.0
    np.testing.assert_allclose(swap_teams(swap_teams(df)).to_numpy(), df.to_numpy(), equal_nan=True)
    sw = swap_teams(df)
    np.testing.assert_allclose(sw["blk_back_chg75_a"], df["blk_back_chg75_h"], equal_nan=True)


def test_dynamic_viterbi_switches_where_the_detector_fires():
    from hackbarca.decode.viterbi import dynamic_transition_costs, transition_costs
    from hackbarca.models.change import change_targets
    from hackbarca.models.l3 import DecodeParams, decode
    n = 200
    proba = np.full((n, 3), [0.6, 0.35, 0.05])          # el modelo duda: H algo mas probable
    proba[100:140] = [0.35, 0.6, 0.05]                  # 40 frames de evidencia debil de A
    runs = [(0, n)]
    params = DecodeParams(temperature=1.0, switch=12.0, to_dead=10.0, from_dead=10.0)
    fixed = decode(proba, runs, params)
    assert (fixed == 0).all()                           # con coste fijo no compensa cambiar
    p_change = np.full((n, 2), 0.05)
    p_change[98:103, 0] = p_change[138:143, 0] = 0.99   # el detector ve el robo y la vuelta
    dyn = decode(proba, runs, DecodeParams(1.0, 12.0, 10.0, 10.0, alpha_switch=1.0), p_change)
    assert (dyn[105:135] == 1).all() and (dyn[:95] == 0).all() and (dyn[150:] == 0).all()
    assert np.allclose(dynamic_transition_costs(12, 10, 10, np.full((3, 2), 0.5), 1.0, 1.0)[0],
                       transition_costs(12, 10, 10))   # p = 0,5 -> coste base
    team, dead = change_targets(np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int8), near=3)
    assert team[17:24].all() and not team[:16].any() and dead[37:44].all() and not dead[:30].any()
