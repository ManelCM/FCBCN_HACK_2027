"""Evaluacion sobre el banco externo: un predictor contra varios partidos, por partido y agregado.

El agregado concatena los partidos separados por un frame X, para que possession_metrics los trate
como mitades independientes (valid_runs) y el spellF1 sea el de todas las posesiones juntas.
"""

from __future__ import annotations

import numpy as np

from hackbarca.eval.evaluate import pm


def evaluate_codes(gt: np.ndarray, pred: np.ndarray, bootstrap_n: int = 0) -> dict:
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)


def pooled(gts: list[np.ndarray], preds: list[np.ndarray], bootstrap_n: int = 0) -> dict:
    sep = np.array([pm.X], dtype=np.int8)
    gt = np.concatenate([np.concatenate([g, sep]) for g in gts])
    pred = np.concatenate([np.concatenate([p, sep]) for p in preds])
    return evaluate_codes(gt, pred, bootstrap_n)


def headline(result: dict) -> str:
    return "spellF1={:.4f} macroF1={:.4f} poss={} (gt {})".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"])
