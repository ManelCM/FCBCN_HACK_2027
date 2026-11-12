"""Por que en el reto salen tantos D cortos con el balon visible: atribucion agregada (ADR-0004).

    scripts/run_aggregate.sh short_dead uv run python -m hackbarca.pipeline.short_dead_report

real_vs_bench (E041) mide 0,68 tramos D de menos de 5 s por minuto en el reto, frente a 0,09 en la GT
del banco y 0,12 en nuestra prediccion del banco, y el 90 % con el balon visible. Este script toma
las features del reto (`make_submission --dump-proba`) y el modelo de la capa 1 del artefacto y
compara tres grupos de frames del reto:
  S = dentro de un D corto (< 5 s) entre dos tramos vivos, con el balon visible;
  V = juego vivo con el balon visible (muestra);
  L = dentro de un D largo (>= 5 s) (muestra).
Para cada feature da su contribucion media al logit de D (pred_contrib de LightGBM) en S menos en V:
las que mas empujan hacia D en los D cortos. Para esas features compara ademas su valor tipificado
en el reto (S, V, L) con el del banco (GT viva y GT D): si en el reto S se parece a L y en el banco
lo vivo no se parece a D, la feature tiene un desajuste de dominio.

Imprime solo medias por grupo y recuentos; ni frames ni filas.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from hackbarca.eval.evaluate import pm
from hackbarca.models.l3 import swap_teams
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.tune_l3 import bench_feature_paths, read_features

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, A, D, X = 0, 1, 2, 3
SHORT = 125   # frames: 5 s
SAMPLE = 1500   # frames por grupo: TreeSHAP sobre 1.200 arboles de 127 hojas es caro
FEATURES = "touch+ball+players+blocks+structure+restart"


def frame_groups(codes: np.ndarray, known: np.ndarray, runs, rng) -> dict:
    """Indices de los grupos S (D corto entre vivos, balon visible), V (vivo visible) y L (D largo)."""
    s_idx, l_idx, v_mask = [], [], np.zeros(len(codes), dtype=bool)
    for a, b in runs:
        segs = pm.segments(codes, [(a, b)])
        for n, (s, e, c) in enumerate(segs):
            if c != D:
                v_mask[s:e] = True
                continue
            if e - s < SHORT and 0 < n < len(segs) - 1:
                s_idx.append(np.arange(s, e))
            elif e - s >= SHORT:
                l_idx.append(np.arange(s, e))
    s_idx = np.concatenate(s_idx) if s_idx else np.zeros(0, dtype=int)
    s_idx = s_idx[known[s_idx]]
    l_idx = np.concatenate(l_idx) if l_idx else np.zeros(0, dtype=int)
    v_idx = np.flatnonzero(v_mask & known)

    def sample(idx):
        return np.sort(rng.choice(idx, min(SAMPLE, len(idx)), replace=False)) if len(idx) else idx

    return dict(S=sample(s_idx), V=sample(v_idx), L=sample(l_idx))


def d_contrib(model, df: pd.DataFrame) -> np.ndarray:
    """Contribucion de cada feature al logit de D, simetrizada como predict_proba (media del
    partido tal cual y con los equipos intercambiados; D no cambia de indice)."""
    n_feat = df.shape[1]
    out = 0
    for x in (df, swap_teams(df)):
        contrib = model.predict(x, pred_contrib=True)
        out = out + contrib[:, 2 * (n_feat + 1):3 * (n_feat + 1) - 1]
    return out / 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=os.path.join("data", "derived", "proba_l3_features.npz"))
    parser.add_argument("--proba", default=os.path.join("data", "derived", "proba_l3.npz"))
    parser.add_argument("--team-name", default="La MasIA")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)

    from hackbarca.pipeline.train_l3_final import load_model
    model = load_model()
    blob = np.load(args.features)
    df = pd.DataFrame(blob["values"], columns=[str(c) for c in blob["columns"]])
    df = df[model.feature_name()]
    known = np.load(args.proba)["ball_known"]
    codes = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", args.team_name + ".csv")))
    rng = np.random.default_rng(0)
    groups = frame_groups(codes, known, [tuple(r) for r in GRADED], rng)
    print("frames por grupo en el reto: " + ", ".join("%s %d" % (k, len(v)) for k, v in groups.items()))

    contrib = {k: d_contrib(model, df.iloc[v]).mean(0) for k, v in groups.items() if len(v)}
    total = {k: float(c.sum()) for k, c in contrib.items()}
    print("contribucion total media al logit de D: " + ", ".join("%s %+.2f" % (k, v) for k, v in total.items()))
    diff = contrib["S"] - contrib["V"]
    top = np.argsort(-diff)[:args.top]

    # valores tipificados: reto (por su desviacion global) y banco (GT viva frente a GT D)
    names = [df.columns[i] for i in top]
    mu, sd = df[names].mean(), df[names].std() + 1e-9
    real_z = {k: ((df[names].iloc[v] - mu) / sd).mean() for k, v in groups.items()}
    live_vals, dead_vals = [], []
    for path, gt in bench_feature_paths(use_tags=False, version=FEATURES,
                                        names=__import__("hackbarca.pipeline.bench_config",
                                                         fromlist=["BENCH_GAMES_V3"]).BENCH_GAMES_V3):
        bdf = read_features(path)[names]
        live = np.flatnonzero((gt == H) | (gt == A))
        dead = np.flatnonzero(gt == D)
        live_vals.append(bdf.iloc[rng.choice(live, min(5000, len(live)), replace=False)])
        dead_vals.append(bdf.iloc[rng.choice(dead, min(5000, len(dead)), replace=False)])
    blive, bdead = pd.concat(live_vals), pd.concat(dead_vals)
    bmu, bsd = pd.concat([blive, bdead]).mean(), pd.concat([blive, bdead]).std() + 1e-9

    print("\nfeatures que mas empujan hacia D en los D cortos del reto (contribucion S - V al logit de D)")
    print("{:<34}{:>9}{:>9}{:>9} | {:>7}{:>7}{:>7} | {:>9}{:>9}".format(
        "feature", "S-V", "S", "V", "z S", "z V", "z L", "banco viv", "banco D"))
    for i, name in zip(top, names):
        print("{:<34}{:>+9.3f}{:>+9.3f}{:>+9.3f} | {:>+7.2f}{:>+7.2f}{:>+7.2f} | {:>+9.2f}{:>+9.2f}".format(
            name, diff[i], contrib["S"][i], contrib["V"][i], real_z["S"][name], real_z["V"][name],
            real_z["L"][name], ((blive[name] - bmu[name]) / bsd[name]).mean(),
            ((bdead[name] - bmu[name]) / bsd[name]).mean()))
    # por familia (prefijo de la feature)
    fam = pd.Series(diff, index=df.columns).groupby(lambda c: c.split("_")[0]).sum().sort_values(ascending=False)
    print("\nS - V por prefijo de feature: " + ", ".join("%s %+.2f" % (k, v) for k, v in fam.head(12).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
