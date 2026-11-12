"""Detector de cambio (E032): probabilidad de que cerca de cada frame cambie la posesion o el balon
pase a estar vivo o muerto. Lo usa el Viterbi para abaratar los cambios donde hay senales de robo o
de paron (costes de transicion dependientes del contexto, como en PathCRF).

Dos clasificadores binarios LightGBM sobre las mismas features que la capa 1:
- cambio de equipo: hay un H<->A a <= NEAR frames;
- cambio vivo/muerto: hay una transicion hacia o desde D a <= NEAR frames.
La prediccion se simetriza con los equipos intercambiados (un cambio no depende de quien sea local).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from hackbarca.features.frame_features import swap_teams

NEAR = 12   # frames (~0,5 s) alrededor de una transicion de la GT
SUBSAMPLE = 5
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
              seed=0, deterministic=True, force_col_wise=True, num_threads=4)
N_ROUNDS = 200


def change_targets(gt: np.ndarray, near: int = NEAR):
    """(team_change, dead_change) por frame, solo dentro de los tramos con GT (codigo 3 = X)."""
    n = len(gt)
    team = np.zeros(n, dtype=np.int8)
    dead = np.zeros(n, dtype=np.int8)
    for t in np.flatnonzero(gt[1:] != gt[:-1]) + 1:
        a, b = gt[t - 1], gt[t]
        if a == 3 or b == 3:
            continue
        lo, hi = max(0, t - near), min(n, t + near + 1)
        if {a, b} == {0, 1}:
            team[lo:hi] = 1
        elif 2 in (a, b):
            dead[lo:hi] = 1
    return team, dead


def train_change(features: list[pd.DataFrame], labels: list[np.ndarray]):
    """(modelo de cambio de equipo, modelo de cambio vivo/muerto)."""
    xs, team_y, dead_y = [], [], []
    for df, gt in zip(features, labels):
        keep = np.flatnonzero(gt != 3)[::SUBSAMPLE]
        team, dead = change_targets(gt)
        xs += [df.iloc[keep], swap_teams(df.iloc[keep])]
        team_y += [team[keep], team[keep]]
        dead_y += [dead[keep], dead[keep]]
    x = pd.concat(xs, ignore_index=True)
    models = []
    for y in (np.concatenate(team_y), np.concatenate(dead_y)):
        models.append(lgb.train(PARAMS, lgb.Dataset(x, y), num_boost_round=N_ROUNDS))
    return tuple(models)


def predict_change(models, df: pd.DataFrame) -> np.ndarray:
    """(n, 2): probabilidad de cambio de equipo y de cambio vivo/muerto, simetrizada."""
    swapped = swap_teams(df)
    return np.column_stack([(m.predict(df) + m.predict(swapped)) / 2 for m in models])
