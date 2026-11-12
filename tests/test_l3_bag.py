import numpy as np
import pandas as pd

from hackbarca.models import l3


def _toy(seed):
    rng = np.random.default_rng(seed)
    n = 3000
    y = rng.integers(0, 3, n)
    df = pd.DataFrame({"x_h": y + rng.normal(0, 0.5, n), "x_a": (y == 1) + rng.normal(0, 0.5, n)})
    return df, y


def test_bag_matches_single_when_one(monkeypatch):
    monkeypatch.setattr(l3, "N_ROUNDS", 20)
    feats, labels = zip(*(_toy(s) for s in range(2)))
    monkeypatch.setattr(l3, "BAGS", 1)
    single = l3.train(list(feats), list(labels))
    assert not isinstance(single, l3.Bag)
    monkeypatch.setattr(l3, "BAGS", 3)
    bag = l3.train(list(feats), list(labels))
    assert isinstance(bag, l3.Bag) and len(bag.boosters) == 3
    p = l3.predict_proba(bag, feats[0])
    assert p.shape == (3000, 3) and np.allclose(p.sum(1), 1)
    np.testing.assert_allclose(bag.boosters[0].predict(feats[0]), single.predict(feats[0]))


def test_xgb_mix(monkeypatch):
    monkeypatch.setattr(l3, "N_ROUNDS", 20)
    monkeypatch.setattr(l3, "BAGS", 1)
    monkeypatch.setattr(l3, "XGB_MIX", True)
    feats, labels = zip(*(_toy(s) for s in range(2)))
    model = l3.train(list(feats), list(labels))
    assert isinstance(model, l3.Bag) and isinstance(model.boosters[1], l3.XgbModel)
    p = l3.predict_proba(model, feats[0])
    assert p.shape == (3000, 3) and np.allclose(p.sum(1), 1)
    assert len(model.feature_importance("gain")) == 2


def test_bundle_roundtrip(monkeypatch, tmp_path):
    from hackbarca.pipeline import train_l3_final as final
    monkeypatch.setattr(l3, "N_ROUNDS", 20)
    monkeypatch.setattr(l3, "BAGS", 2)
    monkeypatch.setattr(l3, "XGB_MIX", True)
    feats, labels = zip(*(_toy(s) for s in range(2)))
    model = l3.train(list(feats), list(labels))
    path = str(tmp_path / "m.txt.gz")
    final.save_model(model, path)
    first = open(path, "rb").read()
    final.save_model(final.load_model(path), path)
    assert open(path, "rb").read() == first      # identico byte a byte
    np.testing.assert_allclose(final.load_model(path).predict(feats[0]), model.predict(feats[0]), rtol=1e-6)
