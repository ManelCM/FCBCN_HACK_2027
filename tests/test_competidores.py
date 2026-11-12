"""Ideas de otros equipos (docs/investigacion/07_competidores.md): E111 (Viterbi por lotes), E113 (pesos por
posesion), E114 (SkillCorner solo con H/A) y E115 (dos cabezas)."""

import numpy as np
import pandas as pd
import pytest

from hackbarca.eval.bench import pm
from hackbarca.models import l3


def _toy(seed, n=3000):
    rng = np.random.default_rng(seed)
    y = np.repeat(rng.integers(0, 3, n // 50), 50)[:n]
    df = pd.DataFrame({"x_h": (y == 0) + rng.normal(0, 0.5, n), "x_a": (y == 1) + rng.normal(0, 0.5, n),
                       "dead": (y == 2) + rng.normal(0, 0.5, n)})
    return df, y


def test_spell_weights_balance_and_clip():
    gt = np.array([0] * 50 + [1] * 500 + [2] * 100 + [0] * 200 + [3] * 10 + [1] * 5, dtype=np.int8)
    w = l3.spell_weights(gt)
    live = (gt == 0) | (gt == 1)
    assert np.isclose(w[live].sum(), live.sum())               # el peso total vivo no cambia
    assert np.all(w[gt == 2] == 1) and np.all(w[gt == 3] == 1)
    assert w[0] > w[100] > 0                                   # la posesion corta pesa mas por frame
    raw = np.clip(8 * 25 / np.array([50, 500, 200, 5]), 0.25, 4)
    ratio = w[[0, 100, 700, len(gt) - 1]] / raw
    assert np.allclose(ratio, ratio[0])                        # recorte [0,25, 4] antes de reescalar


def test_spell_weights_used_by_training(monkeypatch):
    monkeypatch.setattr(l3, "N_ROUNDS", 10)
    monkeypatch.setattr(l3, "BAGS", 1)
    feats, labels = zip(*(_toy(s) for s in range(2)))
    plain = l3.train(list(feats), list(labels))
    weighted = l3.train(list(feats), list(labels), [l3.spell_weights(y) for y in labels])
    assert not np.allclose(plain.predict(feats[0]), weighted.predict(feats[0]))


def test_two_heads_shape_symmetry_and_roundtrip(monkeypatch, tmp_path):
    from hackbarca.features.frame_features import swap_teams
    from hackbarca.pipeline import train_l3_final as final
    monkeypatch.setattr(l3, "N_ROUNDS", 10)
    monkeypatch.setattr(l3, "BAGS", 1)
    monkeypatch.setattr(l3, "HEADS", 2)
    feats, labels = zip(*(_toy(s) for s in range(3)))
    model = l3.train(list(feats), list(labels), team_only=[False, False, True])
    assert isinstance(model, l3.TwoHeads)
    p = l3.predict_proba(model, feats[0])
    assert p.shape == (3000, 3) and np.allclose(p.sum(1), 1)
    q = l3.predict_proba(model, swap_teams(feats[0]))
    np.testing.assert_allclose(q, p[:, [1, 0, 2]], atol=1e-9)   # simetria por intercambio de equipos
    acc = np.mean(np.argmax(p, 1) == labels[0])
    assert acc > 0.6
    path = str(tmp_path / "m.txt.gz")
    final.save_model(model, path)
    np.testing.assert_allclose(final.load_model(path).predict(feats[0]), model.predict(feats[0]), rtol=1e-6)
    stage1, stage2 = l3.train_stacked(list(feats), list(labels), team_only=[False, False, True],
                                      weights=[l3.spell_weights(y) for y in labels])
    assert l3.predict_stacked((stage1, stage2), feats[0]).shape == (3000, 3)


def test_team_only_needs_two_heads(monkeypatch):
    monkeypatch.setattr(l3, "N_ROUNDS", 5)
    monkeypatch.setattr(l3, "HEADS", 1)
    feats, labels = zip(*(_toy(s) for s in range(2)))
    with pytest.raises(ValueError):
        l3.train(list(feats), list(labels), team_only=[False, True])


def test_skillcorner_team_only_gt():
    from hackbarca.pipeline.extra_data import team_only_gt
    gt = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    ball = np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(team_only_gt(gt, ball), [0, 3, 3, 3, 0, 1])


def test_decode_batch_matches_decode():
    from hackbarca.pipeline.decode_variants import decode_batch
    rng = np.random.default_rng(0)
    n = 600
    gt = np.repeat(rng.integers(0, 3, n // 30), 30).astype(np.int8)
    p = np.full((n, 3), 0.15)
    p[np.arange(n), gt] = 0.7
    p = np.clip(p + rng.normal(0, 0.15, p.shape), 0.01, None)
    p /= p.sum(1, keepdims=True)
    gt[:20] = 3
    runs = pm.valid_runs(gt)
    cands = [l3.DecodeParams(), l3.DecodeParams(0.5, 5.0, 3.0, 8.0), l3.DecodeParams(2.0, 50.0, 30.0, 40.0)]
    got = decode_batch(p, runs, cands, batch=2)
    for c, params in enumerate(cands):
        np.testing.assert_array_equal(got[c], l3.decode(p, runs, params))


def test_stacked_team_only_at_the_end(monkeypatch):
    """E114: los partidos de SkillCorner (team_only) van al final; ningun pliegue interno puede quedarse
    solo con ellos."""
    monkeypatch.setattr(l3, "N_ROUNDS", 5)
    monkeypatch.setattr(l3, "BAGS", 1)
    monkeypatch.setattr(l3, "HEADS", 2)
    feats, labels = zip(*(_toy(s, n=600) for s in range(7)))
    flags = [False] * 3 + [True] * 4
    stage1, stage2 = l3.train_stacked(list(feats), list(labels), team_only=flags)
    assert l3.predict_stacked((stage1, stage2), feats[0]).shape == (600, 3)


def test_carry_reveal_walks_then_waits():
    """E123: el balón muerto revelado espera, se lleva andando y se queda quieto en el punto de saque."""
    from hackbarca.io.dead_carry import carry_reveal
    n = 1000
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bx[:100], by[:100] = 10.0, 20.0
    bx[900:], by[900:] = 40.0, 20.0
    missing = np.isnan(bx)
    carry_reveal(np.random.default_rng(0), bx, by, missing, prob=1.0, noise=0.0)
    assert np.all(np.isfinite(bx))
    speed = np.hypot(np.diff(bx), np.diff(by)) * 25
    assert speed[99:900].max() < 4.0                       # paso de jugador (+ vaivén), nunca un pase
    assert np.allclose(bx[100:110], 10.0, atol=0.5)       # primero quieto donde quedó
    assert np.allclose(bx[850:900], 40.0, atol=1e-6)      # al final, quieto en el punto de saque
    short_x = np.array([0.0] + [np.nan] * 10 + [30.0])    # hueco corto: interpolación de siempre
    short_y = np.zeros(12)
    carry_reveal(np.random.default_rng(0), short_x, short_y, np.isnan(short_x), prob=1.0, noise=0.0)
    np.testing.assert_allclose(short_x, np.linspace(0, 30, 12))
