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
BAGS = 1       # M5e (E093): modelos por capa con desfases distintos del submuestreo, promediados
XGB_MIX = False  # M5d (E096): cada capa promedia LightGBM con un XGBoost de capacidad parecida
XGB_PARAMS = dict(objective="multi:softprob", num_class=3, eta=0.05, grow_policy="lossguide", max_leaves=127,
                  max_depth=0, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", nthread=4, seed=SEED)
# E115 (I4, katenaball/FCB-Hacks): dos cabezas binarias, P(vivo) y P(local | vivo), en vez de un multiclase
HEADS = 1
# E113 (I2, FCB-Hacks): peso de un frame vivo = SPELL_REF_S / duracion de su posesion, recortado
SPELL_REF_S = 8.0
SPELL_CLIP = (0.25, 4.0)


def spell_weights(gt: np.ndarray, fps: int = 25, ref_s: float = SPELL_REF_S, clip=SPELL_CLIP) -> np.ndarray:
    """Pesos por frame para que cada posesion cuente parecido, como en spellF1.

    Cada racha maxima de H o de A (un D o un X la corta) pesa ref_s / su duracion por frame, recortado a
    `clip`. Despues se reescalan los frames vivos para que su peso total sea su numero de frames: el
    equilibrio entre vivo y D no cambia. D y X pesan 1."""
    gt = np.asarray(gt)
    w = np.ones(len(gt))
    if not len(gt):
        return w
    edges = np.flatnonzero(np.diff(gt.astype(np.int16)) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(gt)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if gt[a] in (0, 1):
            w[a:b] = np.clip(ref_s * fps / (b - a), *clip)
    live = (gt == 0) | (gt == 1)
    if live.any():
        w[live] *= live.sum() / w[live].sum()
    return w


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


class Bag:
    """Media de varios Booster (M5e): misma interfaz que lgb.Booster para predecir."""

    def __init__(self, boosters):
        self.boosters = list(boosters)

    def predict(self, df):
        return np.mean([b.predict(df) for b in self.boosters], axis=0)

    def feature_importance(self, kind: str = "split"):
        return np.sum([b.feature_importance(kind) for b in self.boosters], axis=0)

    def feature_name(self):
        return self.boosters[0].feature_name()


class XgbModel:
    """XGBoost con la interfaz de lgb.Booster que usa el pipeline (M5d)."""

    def __init__(self, booster, names):
        self.booster, self.names = booster, list(names)

    def predict(self, df):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(df[self.names]))

    def feature_importance(self, kind: str = "split"):
        score = self.booster.get_score(importance_type="gain" if kind == "gain" else "weight")
        return np.array([score.get(n, 0.0) for n in self.names])

    def feature_name(self):
        return self.names


class TwoHeads:
    """P(vivo) y P(local | vivo) (E115) con la interfaz de lgb.Booster: predict da (n, 3) H/A/D."""

    def __init__(self, alive, home):
        self.alive, self.home = alive, home

    def predict(self, df):
        pa, ph = self.alive.predict(df), self.home.predict(df)
        return np.stack([pa * ph, pa * (1 - ph), 1 - pa], axis=1)

    def feature_importance(self, kind: str = "split"):
        return self.alive.feature_importance(kind) + self.home.feature_importance(kind)

    def feature_name(self):
        return self.alive.feature_name()


def model_to_dict(model) -> dict:
    """Bag / XgbModel / TwoHeads / lgb.Booster a un dict de texto, para guardarlo en el artefacto."""
    if isinstance(model, TwoHeads):
        return {"two_heads": [model_to_dict(model.alive), model_to_dict(model.home)]}
    if isinstance(model, Bag):
        return {"bag": [model_to_dict(m) for m in model.boosters]}
    if isinstance(model, XgbModel):
        return {"xgb": bytes(model.booster.save_raw("json")).decode(), "names": model.names}
    return {"lgb": model.model_to_string().replace("\r\n", "\n")}


def model_from_dict(data: dict):
    if "two_heads" in data:
        return TwoHeads(*(model_from_dict(m) for m in data["two_heads"]))
    if "bag" in data:
        return Bag(model_from_dict(m) for m in data["bag"])
    if "xgb" in data:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(bytearray(data["xgb"].encode()))
        return XgbModel(booster, data["names"])
    return lgb.Booster(model_str=data["lgb"])


def _train_two_heads(x, y, w, team_only, offset: int):
    """E115: P(vivo) con las filas de partidos con D (no las de team_only) y P(local) con las filas H/A."""
    params = {k: v for k, v in LGB_PARAMS.items() if k != "num_class"}
    params.update(objective="binary", seed=LGB_PARAMS["seed"] + offset)
    rows = ~team_only
    if not rows.any():
        raise ValueError("cabeza de vivo sin filas: todos los partidos del entrenamiento son team_only")
    alive = lgb.train(params, lgb.Dataset(x[rows], (y[rows] != 2).astype(int),
                                          weight=None if w is None else w[rows]), num_boost_round=N_ROUNDS)
    live = y != 2
    home = lgb.train(params, lgb.Dataset(x[live], (y[live] == 0).astype(int),
                                         weight=None if w is None else w[live]), num_boost_round=N_ROUNDS)
    return TwoHeads(alive, home)


def _train_one(features, labels, offset: int, weights=None, team_only=None):
    xs, ys, ws, ts = [], [], [], []
    for i, (df, y) in enumerate(zip(features, labels)):
        keep = np.flatnonzero(y != 3)[offset::SUBSAMPLE]
        xs += [df.iloc[keep], swap_teams(df.iloc[keep])]
        ys += [y[keep], _swap_labels(y[keep])]
        if weights is not None:
            ws += [weights[i][keep]] * 2
        flag = bool(team_only[i]) if team_only is not None else False
        ts += [np.full(len(keep), flag)] * 2
    x, y = pd.concat(xs, ignore_index=True), np.concatenate(ys)
    w = np.concatenate(ws) if weights is not None else None
    if HEADS == 2:
        return _train_two_heads(x, y, w, np.concatenate(ts), offset)
    if team_only is not None and any(team_only):
        raise ValueError("team_only (filas solo de equipo) necesita HEADS = 2")
    model = lgb.train(dict(LGB_PARAMS, seed=LGB_PARAMS["seed"] + offset), lgb.Dataset(x, y, weight=w),
                      num_boost_round=N_ROUNDS)
    if not XGB_MIX:
        return model
    import xgboost as xgb
    other = xgb.train(dict(XGB_PARAMS, seed=XGB_PARAMS["seed"] + offset), xgb.DMatrix(x, y, weight=w),
                      num_boost_round=N_ROUNDS)
    return Bag([model, XgbModel(other, x.columns)])


def train(features: list[pd.DataFrame], labels: list[np.ndarray], weights=None, team_only=None):
    """features/labels por partido; labels 0/1/2 (H/A/D) y 3 (X, se descarta). Con BAGS > 1, un Bag
    de modelos con desfases repartidos en el submuestreo (y semillas distintas).

    weights: pesos por frame de cada partido (E113, `spell_weights`) o None.
    team_only: por partido, True si sus filas solo valen para la cabeza de equipo (E114/E115, HEADS = 2)."""
    if BAGS == 1:
        return _train_one(features, labels, 0, weights, team_only)
    return Bag(_train_one(features, labels, round(k * SUBSAMPLE / BAGS), weights, team_only) for k in range(BAGS))


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


def train_stacked(features: list[pd.DataFrame], labels: list[np.ndarray], inner_folds: int = INNER_FOLDS,
                  weights=None, team_only=None):
    """(capa1, capa2). La capa 2 se entrena con probabilidades de capa 1 fuera de muestra
    (cada grupo de partidos predicho por un modelo entrenado con los demas). weights / team_only como
    en `train`, por partido, en las dos capas."""
    n = len(features)

    def pick(xs, idx):
        return None if xs is None else [xs[j] for j in idx]

    inner = [None] * n
    k = min(inner_folds, n)
    if team_only is not None and any(team_only):
        # E114: los partidos solo de equipo van al final; con bloques contiguos un pliegue interno se
        # entrenaria solo con ellos y la cabeza de vivo se quedaria sin filas. Se intercalan.
        groups = [np.arange(n)[g::k] for g in range(k)]
    else:
        groups = np.array_split(np.arange(n), k)   # como siempre: E058b/E110 se reproducen igual
    for group in groups:
        rest = [j for j in range(n) if j not in group]
        m = train([features[j] for j in rest], [labels[j] for j in rest], pick(weights, rest), pick(team_only, rest))
        for i in group:
            inner[i] = _stack_input(features[i], predict_proba(m, features[i]))
    stage2 = train(inner, labels, weights, team_only)
    stage1 = train(features, labels, weights, team_only)
    return stage1, stage2


def predict_stacked(models, df: pd.DataFrame) -> np.ndarray:
    stage1, stage2 = models
    return predict_proba(stage2, _stack_input(df, predict_proba(stage1, df)))
