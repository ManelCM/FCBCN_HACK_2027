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
