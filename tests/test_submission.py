import numpy as np

from hackbarca.eval.evaluate import validate_csv
from hackbarca.models.l1 import L1Params, apply_tags
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.make_submission import write_csv


def test_write_csv_x_only_outside_graded(tmp_path):
    codes = np.zeros(199319, dtype=np.int8)
    codes[50000:50100] = 2
    path = tmp_path / "team.csv"
    write_csv(str(path), codes)
    result = validate_csv(str(path))
    assert result.ok, result.reason
    graded = sum(stop - start for start, stop in GRADED)
    assert result.label_counts["X"] == 199319 - graded
    assert result.label_counts["D"] == 100


def test_apply_tags_raises_dead_before_and_inside_window():
    p_dead = np.full(1000, 0.05)
    params = L1Params(tag_before=100, p_tag_before=0.9, p_tag_in=0.4)
    out = apply_tags(p_dead, [(500, 625)], params)
    assert np.allclose(out[400:500], 0.9)
    assert np.allclose(out[500:626], 0.4)
    assert np.allclose(out[:400], 0.05) and np.allclose(out[626:], 0.05)
    assert apply_tags(p_dead, [(500, 625)], L1Params()) is p_dead


def test_stacked_model_roundtrip(tmp_path):
    import pandas as pd
    from hackbarca.models import l3
    from hackbarca.pipeline.train_l3_final import load_model, save_model
    rng = np.random.default_rng(0)
    old = dict(l3.LGB_PARAMS), l3.N_ROUNDS
    l3.LGB_PARAMS.update(num_leaves=7, min_data_in_leaf=20)
    l3.N_ROUNDS = 10
    try:
        feats, labels = [], []
        for _ in range(3):
            n = 3000
            y = (np.arange(n) // 300) % 3
            df = pd.DataFrame({"d1_h": rng.normal(y == 0, 1, n), "d1_a": rng.normal(y == 1, 1, n),
                               "near_sign": rng.normal(0, 1, n)}).astype(np.float32)
            feats.append(df)
            labels.append(y.astype(np.int8))
        models = l3.train_stacked(feats, labels)
        p = l3.predict_stacked(models, feats[0])
        save_model(models[0], str(tmp_path / "s1.txt.gz"))
        save_model(models[1], str(tmp_path / "s2.txt.gz"))
        loaded = (load_model(str(tmp_path / "s1.txt.gz")), load_model(str(tmp_path / "s2.txt.gz")))
        np.testing.assert_allclose(l3.predict_stacked(loaded, feats[0]), p)
        assert p.shape == (3000, 3) and np.allclose(p.sum(1), 1)
    finally:
        l3.LGB_PARAMS.clear()
        l3.LGB_PARAMS.update(old[0])
        l3.N_ROUNDS = old[1]


def test_training_is_byte_deterministic():
    # E024: sin force_col_wise, dos entrenamientos con distinta carga daban artefactos distintos
    import pandas as pd
    from hackbarca.models import l3
    assert l3.LGB_PARAMS.get("deterministic") and l3.LGB_PARAMS.get("force_col_wise")
    rng = np.random.default_rng(1)
    n = 4000
    y = ((np.arange(n) // 400) % 3).astype(np.int8)
    df = pd.DataFrame({"d1_h": rng.normal(y == 0, 1, n), "d1_a": rng.normal(y == 1, 1, n),
                       "near_sign": rng.normal(0, 1, n)}).astype(np.float32)
    old = dict(l3.LGB_PARAMS), l3.N_ROUNDS
    l3.LGB_PARAMS.update(num_leaves=15, min_data_in_leaf=20)
    l3.N_ROUNDS = 20
    try:
        first = l3.train([df], [y]).model_to_string()
        second = l3.train([df], [y]).model_to_string()
    finally:
        l3.LGB_PARAMS.clear()
        l3.LGB_PARAMS.update(old[0])
        l3.N_ROUNDS = old[1]
    assert first == second
