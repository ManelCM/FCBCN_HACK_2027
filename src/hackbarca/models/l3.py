"""Modelo L3: LightGBM multiclase por frame (H/A/D) + Viterbi con coste de cambio.

Entrenado solo en el banco externo. Aumento de datos intercambiando equipos (H<->A) y prediccion
simetrizada: p(x) promediada con la de x intercambiado, para no heredar un sesgo de local.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import lightgbm as lgb
import numpy as np
import pandas as pd

from hackbarca.decode.viterbi import decode_runs, transition_costs
from hackbarca.features.frame_features import frame_features, swap_teams
from hackbarca.io.load_match import MatchTracking

SEED = 0
# E010: mas capacidad mejora de forma monotona en el banco (7 hojas 0,455 ... 127 hojas 0,502)
LGB_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=127,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=SEED, deterministic=True, num_threads=4)
N_ROUNDS = 400
SUBSAMPLE = 5  # un frame de cada 5 para entrenar: los vecinos son casi identicos


@dataclass(frozen=True)
class DecodeParams:
    temperature: float = 1.0   # escala de las emisiones (frames correlados: <1 = menos confianza)
    switch: float = 20.0
    to_dead: float = 10.0
    from_dead: float = 10.0


def ensure_period(match: MatchTracking, halftime_frame: int | None = None) -> MatchTracking:
    """Si el tracking no trae periodo (ATD), lo deduce de la frontera del descanso."""
    if match.period is not None:
        return match
    if halftime_frame is None:
        halftime_frame = match.n_frames // 2
    period = np.where(np.arange(match.n_frames) < halftime_frame, 1, 2).astype(np.int8)
    return replace(match, period=period)


def _swap_labels(y: np.ndarray) -> np.ndarray:
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def train(features: list[pd.DataFrame], labels: list[np.ndarray]) -> lgb.Booster:
    """features/labels por partido; labels 0/1/2 (H/A/D) y 3 (X, se descarta)."""
    xs, ys = [], []
    for df, y in zip(features, labels):
        keep = np.flatnonzero(y != 3)[::SUBSAMPLE]
        xs += [df.iloc[keep], swap_teams(df.iloc[keep])]
        ys += [y[keep], _swap_labels(y[keep])]
    data = lgb.Dataset(pd.concat(xs, ignore_index=True), np.concatenate(ys))
    return lgb.train(LGB_PARAMS, data, num_boost_round=N_ROUNDS)


def predict_proba(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    p = model.predict(df)
    q = model.predict(swap_teams(df))
    sym = (p + q[:, [1, 0, 2]]) / 2
    return sym / sym.sum(1, keepdims=True)


def decode(proba: np.ndarray, runs, params: DecodeParams = DecodeParams()) -> np.ndarray:
    cost = -params.temperature * np.log(proba + 1e-6)
    return decode_runs(cost, transition_costs(params.switch, params.to_dead, params.from_dead), runs)


def features_for(match: MatchTracking, windows=None, halftime_frame=None, version: int = 1) -> pd.DataFrame:
    return frame_features(ensure_period(match, halftime_frame), windows, version=version)
