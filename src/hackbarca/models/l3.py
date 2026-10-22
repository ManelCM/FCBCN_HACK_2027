"""Modelo L3: LightGBM multiclase por frame (H/A/D) + Viterbi con coste de cambio.

Entrenado solo en el banco externo. Aumento de datos intercambiando equipos (H<->A) y prediccion
simetrizada: p(x) promediada con la de x intercambiado, para no heredar un sesgo de local.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import lightgbm as lgb
import numpy as np
import pandas as pd

from hackbarca.decode.viterbi import decode_runs, dynamic_transition_costs, transition_costs
from hackbarca.features.frame_features import frame_features, swap_teams
from hackbarca.io.load_match import MatchTracking

SEED = 0
# E010: mas capacidad mejora de forma monotona en el banco (7 hojas 0,455 ... 127 hojas 0,502)
LGB_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=127,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=SEED, deterministic=True, num_threads=4,
                  # sin forzarlo, LightGBM elige col/row-wise cronometrando la maquina y dos
                  # entrenamientos con distinta carga dan modelos distintos (E024)
                  force_col_wise=True)
N_ROUNDS = 400
SUBSAMPLE = 5  # un frame de cada 5 para entrenar: los vecinos son casi identicos


@dataclass(frozen=True)
class DecodeParams:
    temperature: float = 1.0   # escala de las emisiones (frames correlados: <1 = menos confianza)
    switch: float = 20.0
    to_dead: float = 10.0
    from_dead: float = 10.0
    alpha_switch: float = 0.0  # E032: cuanto abarata el detector de cambio un H<->A (0 = fijo)
    alpha_dead: float = 0.0    # idem para entrar o salir de D


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


def decode(proba: np.ndarray, runs, params: DecodeParams = DecodeParams(),
           p_change: np.ndarray | None = None) -> np.ndarray:
    """Viterbi por tramo. Con `p_change` (detector de cambio, E032) y alphas > 0, el coste de cada
    transicion varia por frame; sin el, es el Viterbi fijo de siempre."""
    cost = -params.temperature * np.log(proba + 1e-6)
    if p_change is not None and (params.alpha_switch or params.alpha_dead):
        trans = dynamic_transition_costs(params.switch, params.to_dead, params.from_dead, p_change,
                                         params.alpha_switch, params.alpha_dead)
    else:
        trans = transition_costs(params.switch, params.to_dead, params.from_dead)
    return decode_runs(cost, trans, runs)


def features_for(match: MatchTracking, windows=None, halftime_frame=None, version: int = 1) -> pd.DataFrame:
    return frame_features(ensure_period(match, halftime_frame), windows, version=version)


# --- Apilado (E018): segunda capa con contexto temporal largo de las probabilidades ---
CTX_WINDOWS = (25, 50, 125, 250)


def _window_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Media de x en [t+lo, t+hi] (recortado a los bordes)."""
    n = len(x)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    a, b = np.clip(idx + lo, 0, n), np.clip(idx + hi + 1, 0, n)
    return (cs[b] - cs[a]) / np.maximum(b - a, 1)


def context_features(proba: np.ndarray, change: bool = False) -> pd.DataFrame:
    """Probabilidades de la capa 1 resumidas hacia atras, centradas y hacia delante.

    change (features v5, E027): ademas, despues menos antes, una senal explicita de "aqui cambia".
    """
    f = {}
    for k, name in ((0, "h"), (1, "a"), (2, "d")):
        p = proba[:, k]
        f["p1_" + name] = p
        for w in CTX_WINDOWS:
            f["ctxc%d_%s" % (w, name)] = _window_mean(p, -w, w)
            f["ctxb%d_%s" % (w, name)] = _window_mean(p, -w, -1)
            f["ctxf%d_%s" % (w, name)] = _window_mean(p, 1, w)
            if change:
                f["ctxchg%d_%s" % (w, name)] = f["ctxf%d_%s" % (w, name)] - f["ctxb%d_%s" % (w, name)]
    return pd.DataFrame(f)


def _stack_input(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    change = "ball_prog_chg25" in df.columns  # familia "ball": ctxchg en la capa 2
    return pd.concat([df.reset_index(drop=True), context_features(proba, change)], axis=1)


INNER_FOLDS = 3  # con <= 3 partidos equivale a dejar uno fuera (E018); con 10 evita 10 modelos


def train_stacked(features: list[pd.DataFrame], labels: list[np.ndarray], inner_folds: int = INNER_FOLDS):
    """(capa1, capa2). La capa 2 se entrena con probabilidades de capa 1 fuera de muestra
    (cada grupo de partidos predicho por un modelo entrenado con los demas)."""
    n = len(features)
    inner = [None] * n
    for group in np.array_split(np.arange(n), min(inner_folds, n)):
        rest = [j for j in range(n) if j not in group]
        m = train([features[j] for j in rest], [labels[j] for j in rest])
        for i in group:
            inner[i] = _stack_input(features[i], predict_proba(m, features[i]))
    stage2 = train(inner, labels)
    stage1 = train(features, labels)
    return stage1, stage2


def predict_stacked(models, df: pd.DataFrame) -> np.ndarray:
    stage1, stage2 = models
    return predict_proba(stage2, _stack_input(df, predict_proba(stage1, df)))
