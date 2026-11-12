import numpy as np
import pandas as pd

from hackbarca.io.external import event_ball_distance, estimate_lags, place_events
from hackbarca.io.load_match import AWAY, HOME, MatchTracking
from hackbarca.pipeline.failure_analysis import (A, D, H, X, change_masks, drop_short, label_oracle, segment_oracle,
                                                 snap_oracle, spell)


def _codes(*parts):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in parts])


def _game(gt, pred, known=None):
    proba = np.full((len(gt), 3), 0.1)
    proba[np.arange(len(gt)), np.minimum(pred, 2)] = 0.8
    return dict(gt=gt, pred=pred, proba=proba, known=np.ones(len(gt), bool) if known is None else known)


GT = _codes((X, 5), (H, 100), (A, 80), (D, 40), (H, 120), (X, 5))


def test_oracles_full_and_empty_masks():
    pred = _codes((X, 5), (H, 130), (A, 50), (D, 40), (H, 120), (X, 5))
    base = spell([GT], [pred])["mean_f1"]
    assert spell([GT], [GT])["mean_f1"] == 1.0
    assert spell([GT], [np.where(np.zeros(len(GT), bool), GT, pred)])["mean_f1"] == base


def test_segment_and_label_oracles_separate_where_and_who():
    # fronteras perfectas pero A y H intercambiados en un tramo: error de "quien"
    pred = GT.copy()
    pred[105:185] = H
    game = _game(GT, pred)
    # el oraculo de etiqueta (fronteras del modelo) no puede separar H|A porque el modelo no corta
    assert spell([GT], [label_oracle(game)])["mean_f1"] < 1.0
    # con las fronteras de la GT, el argmax medio del modelo da H en el tramo A: sigue mal
    assert spell([GT], [segment_oracle(game)])["mean_f1"] < 1.0
    # frontera desplazada 30 frames con la etiqueta bien: el oraculo de fronteras GT lo arregla
    late = _codes((X, 5), (H, 130), (A, 50), (D, 40), (H, 120), (X, 5))
    assert spell([GT], [segment_oracle(_game(GT, late))])["mean_f1"] == 1.0


def test_snap_oracle_moves_only_close_boundaries():
    late = _codes((X, 5), (H, 110), (A, 70), (D, 40), (H, 120), (X, 5))   # H->A 10 frames tarde
    assert np.array_equal(snap_oracle(_game(GT, late), 12), GT)
    assert np.array_equal(snap_oracle(_game(GT, late), 5), late)


def test_change_masks_counts_missed_and_false():
    missed = _codes((X, 5), (H, 180), (D, 40), (H, 120), (X, 5))          # no ve el A
    known = np.ones(len(GT), bool)
    known[100:110] = False                                                  # el robo cae en un hueco
    seen, gap, false, counts = change_masks(_game(GT, missed, known))
    assert counts["no vistos, en hueco"] == 1 and counts["sobrantes"] == 0
    assert gap[105:185].all() and not seen.any()
    fixed = np.where(gap, GT, missed)
    assert spell([GT], [fixed])["mean_f1"] > spell([GT], [missed])["mean_f1"]
    extra = GT.copy()
    extra[250:262] = A                                                      # cambio falso
    _, _, false, counts = change_masks(_game(GT, extra))
    assert counts["sobrantes"] >= 1 and false[250:262].all()


def test_drop_short_removes_flicker():
    flick = GT.copy()
    flick[40:46] = A
    runs = [(5, len(GT) - 5)]
    assert np.array_equal(drop_short(flick, runs, 12), GT)


def _match(n, ball_x, period):
    return MatchTracking(n_frames=n, fps=25, pitch_length=105.0, pitch_width=68.0,
                         team_of_track=np.array([HOME, AWAY], dtype=np.int8),
                         players_x=np.zeros((n, 2)), players_y=np.zeros((n, 2)),
                         players_speed=np.zeros((n, 2)), ball_x=ball_x, ball_y=np.full(n, 34.0),
                         ball_speed=np.zeros(n), period=period)


def test_place_events_and_lag_recovery():
    n = 3001
    period = np.zeros(n, dtype=np.int8)
    period[:1500], period[1501:] = 1, 2                                      # frame X entre partes
    ball_x = np.full(n, np.nan)
    ball_x[:1500] = np.arange(1500) * 0.05                                   # balon que avanza
    ball_x[1501:] = 100 - np.arange(1500) * 0.05
    true_lag = {1: 30, 2: -20}
    rows = []
    for p, first in ((1, 0), (2, 1501)):
        for t in np.arange(5, 50, 2.0):
            f = first + int(round(t * 25)) + true_lag[p]
            rows.append(dict(team=HOME, type="PASS", subtype="", period=p, t0=t, t1=t + 1,
                             sx=ball_x[f], sy=34.0, ex=np.nan, ey=np.nan))
    table = pd.DataFrame(rows)
    match = _match(n, ball_x, period)
    lags = estimate_lags(table, match)
    assert lags == true_lag
    placed = place_events(table, period, 25, lags)
    assert np.nanmax(event_ball_distance(placed, match)) < 1e-6
    assert (placed["end"] >= placed["start"]).all()


def test_shape_stats_counts_changes_and_gaps():
    from hackbarca.pipeline.real_vs_bench import shape_stats
    codes = _codes((H, 1500), (A, 1500), (D, 1500), (H, 1500))          # 4 min
    known = np.ones(len(codes), bool)
    known[1400:1700] = False                                             # el cambio H->A cae en un hueco largo
    s = shape_stats(codes, [(0, len(codes))], known)
    assert abs(s["minutes"] - 4.0) < 1e-9
    assert abs(s["possessions_per_min"] - 0.75) < 1e-9
    assert abs(s["direct_changes_per_min"] - 0.25) < 1e-9 and abs(s["restarts_per_min"] - 0.25) < 1e-9
    assert s["direct_changes_in_long_gap"] == 1.0
    assert abs(s["dead_share"] - 0.25) < 1e-9


def test_drop_short_dead_merges_split_possession():
    from hackbarca.pipeline.failure_analysis import drop_short_dead
    split = _codes((X, 5), (H, 60), (D, 20), (H, 60), (D, 100), (A, 50), (X, 5))
    out = drop_short_dead(split, [(5, len(split) - 5)], 25)
    assert (out[65:85] == H).all()            # D corto entre dos H -> H
    assert (out[145:245] == D).all()          # D largo se queda


def test_short_dead_groups():
    from hackbarca.pipeline.short_dead_report import frame_groups
    codes = _codes((H, 200), (D, 50), (H, 200), (D, 300), (A, 200))
    known = np.ones(len(codes), bool)
    known[210:220] = False
    g = frame_groups(codes, known, [(0, len(codes))], np.random.default_rng(0))
    assert len(g["S"]) == 40 and g["S"].min() == 200 and g["S"].max() == 249
    assert len(g["L"]) == 300 and (codes[g["V"]] != D).all()


def test_inject_short_dead_rates():
    from hackbarca.pipeline.failure_analysis import inject_short_dead
    pred = _codes(*[(c, 1500) for c in (H, A) * 20])                  # 40 min, 39 cambios H<->A
    runs = [(0, len(pred))]
    out = inject_short_dead(pred, runs, 1.0, 0.5, np.random.default_rng(0))
    from hackbarca.eval.evaluate import pm
    n_dead = sum(1 for a, b, c in pm.segments(out, runs) if c == D)
    assert 40 <= n_dead <= 85                                          # ~40 + ~20 por Poisson
    assert np.array_equal(inject_short_dead(pred, runs, 0, 0, np.random.default_rng(0)), pred)
