"""Tests del wrapper de evaluacion con datos SINTETICOS (ADR-0004)."""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hackbarca.eval.evaluate import (  # noqa: E402
    evaluate_submission, summarize, validate_csv,
)

N = 50  # frames sinteticos


def _write_csv(path, labels):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label"])
        for frame, label in enumerate(labels):
            writer.writerow([frame, label])


def _synthetic_labels():
    # Dos posesiones H y A separadas por D, suficiente para generar spells.
    return (["H"] * 20) + (["D"] * 5) + (["A"] * 20) + (["D"] * 5)


def test_validate_csv_ok(tmp_path):
    path = tmp_path / "ok.csv"
    _write_csv(path, _synthetic_labels())

    result = validate_csv(str(path), expected_rows=N)

    assert result.ok
    assert bool(result) is True
    assert result.n_rows == N
    assert result.label_counts == {"H": 20, "A": 20, "D": 10, "X": 0}
    assert result.n_spells == 2


def test_validate_csv_wrong_row_count(tmp_path):
    path = tmp_path / "bad_rows.csv"
    _write_csv(path, _synthetic_labels()[:-1])  # una fila de menos

    result = validate_csv(str(path), expected_rows=N)

    assert not result.ok
    assert bool(result) is False
    assert "row count" in result.reason


def test_validate_csv_bad_label(tmp_path):
    labels = _synthetic_labels()
    labels[0] = "Z"
    path = tmp_path / "bad_label.csv"
    _write_csv(path, labels)

    result = validate_csv(str(path), expected_rows=N)

    assert not result.ok
    assert "labels outside" in result.reason


def test_validate_csv_empty_label_rejected(tmp_path):
    path = tmp_path / "empty_label.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label"])
        writer.writerow([0, ""])
        for frame in range(1, N):
            writer.writerow([frame, "D"])

    result = validate_csv(str(path), expected_rows=N)
    assert not result.ok
    assert "empty label" in result.reason


def test_evaluate_submission_perfect_match(tmp_path):
    labels = _synthetic_labels()
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    _write_csv(gt_path, labels)
    _write_csv(pred_path, labels)

    result, reason = evaluate_submission(str(pred_path), str(gt_path), bootstrap_n=0)

    assert reason is None
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["spell_f1_mean"] == pytest.approx(1.0)
    summary = summarize(result)
    assert "macroF1=1.0000" in summary
    assert "spellF1=1.0000" in summary


def test_evaluate_submission_invalid_file_returns_reason(tmp_path):
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    _write_csv(gt_path, _synthetic_labels())
    _write_csv(pred_path, _synthetic_labels()[:-5])  # longitud distinta

    result, reason = evaluate_submission(str(pred_path), str(gt_path))

    assert result is None
    assert "row count" in reason


def test_evaluate_submission_with_bootstrap_has_interval(tmp_path):
    labels = _synthetic_labels() * 3  # mas posesiones para que el bootstrap tenga sentido
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    _write_csv(gt_path, labels)
    _write_csv(pred_path, labels)

    result, reason = evaluate_submission(str(pred_path), str(gt_path), bootstrap_n=50)

    assert reason is None
    assert result["spell_bootstrap_low"] is not None
    assert result["spell_bootstrap_high"] is not None
