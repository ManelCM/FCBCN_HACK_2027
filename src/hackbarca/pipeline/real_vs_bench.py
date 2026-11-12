"""Forma de las posesiones en el reto frente al banco: solo agregados (ADR-0004).

    scripts/run_aggregate.sh real_vs_bench uv run python -m hackbarca.pipeline.real_vs_bench \
        --oof <oof del banco v3> [--proba data/derived/proba_l3.npz]

Sin GT del reto no sabemos la nota, pero si como es lo que predecimos: cuantas posesiones por
minuto, cuanto duran, cuantos cambios H<->A directos, cuanto D, en que contexto de balon caen los
cambios y como de seguro esta el modelo. Se compara con la GT del banco (lo que "deberia" salir si
el reto se parece al banco), con nuestra prediccion en el banco (el sesgo conocido del modelo) y con
la GT publica de 5 min. Si en el reto predecimos muchos menos cambios por minuto que la GT del banco
mientras en el banco el modelo se queda cerca, estamos perdiendo robos en el reto.

Imprime solo agregados del partido entero; no imprime frames, filas ni series.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.io.degrade import degrade
from hackbarca.io.external import controlled_possession, is_dfl, load_bench_game
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES_V3
from hackbarca.pipeline.public_window import GT_PATH, WINDOW

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, A, D, X = 0, 1, 2, 3
FPS = 25
LONG_GAP = 100


def long_gaps(known: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate(([0], (~known).astype(np.int8), [0])))
    out = np.zeros(len(known), dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a > LONG_GAP:
            out[a:b] = True
    return out


def shape_stats(codes: np.ndarray, runs, known: np.ndarray | None = None, proba: np.ndarray | None = None) -> dict:
    """Agregados de forma de una secuencia H/A/D dentro de `runs` (listas de [inicio, fin))."""
    frames = sum(b - a for a, b in runs)
    minutes = frames / FPS / 60
    spells = pm.possession_spells(codes, runs)
    dur = np.array([e - s for s, e, _ in spells]) / FPS
    segs = pm.segments(codes, runs)
    dead = np.array([b - a for a, b, c in segs if c == D]) / FPS
    direct = via_dead = 0
    change_frames = []
    for s, e in runs:
        w = codes[s:e]
        idx = np.flatnonzero(w[1:] != w[:-1]) + 1
        for t in idx:
            a, b = int(w[t - 1]), int(w[t])
            if {a, b} == {H, A}:
                direct += 1
                change_frames.append(s + t)
            elif a == D:
                via_dead += 1
    out = dict(minutes=minutes, possessions_per_min=len(spells) / minutes,
               dur_mean=float(dur.mean()) if len(dur) else np.nan,
               **{"dur_p%d" % q: float(np.percentile(dur, q)) if len(dur) else np.nan for q in (10, 25, 50, 75, 90)},
               short_share=float(np.mean(dur < 3)) if len(dur) else np.nan,
               dead_share=float(sum((codes[a:b] == D).sum() for a, b in runs) / frames),
               dead_runs_per_min=len(dead) / minutes,
               dead_p50=float(np.median(dead)) if len(dead) else np.nan,
               direct_changes_per_min=direct / minutes, restarts_per_min=via_dead / minutes)
    if known is not None:
        mask = np.zeros(len(codes), dtype=bool)
        for a, b in runs:
            mask[a:b] = True
        gap = long_gaps(known)
        cf = np.array(change_frames, dtype=int)
        out["long_gap_share"] = float(gap[mask].mean())
        # cuanto D hay dentro de los huecos largos y con el balon visible
        out["dead_share_in_long_gap"] = float(np.mean(codes[mask & gap] == D))
        out["dead_share_visible"] = float(np.mean(codes[mask & known] == D))
        # tramos D: cuantos son cortos (< 5 s) y cuantos empiezan dentro de un hueco largo
        dsegs = [(a, b) for a, b, c in segs if c == D]
        short_d = [(a, b) for a, b in dsegs if b - a < 5 * FPS]
        out["short_dead_runs_per_min"] = len(short_d) / minutes
        out["short_dead_in_long_gap"] = float(np.mean([gap[a] for a, _ in short_d])) if short_d else np.nan
        out["dead_runs_in_long_gap"] = float(np.mean([gap[a] for a, _ in dsegs])) if dsegs else np.nan
        # tramos D cortos entre dos tramos del mismo equipo (parten una posesion) y entre equipos
        # distintos (sustituyen a un cambio directo)
        same = diff = 0
        for n, (a, b, c) in enumerate(segs):
            if c == D and b - a < 5 * FPS and 0 < n < len(segs) - 1 and segs[n - 1][1] == a and segs[n + 1][0] == b:
                same += segs[n - 1][2] == segs[n + 1][2]
                diff += segs[n - 1][2] != segs[n + 1][2]
        out["short_dead_same_team_per_min"] = same / minutes
        out["short_dead_other_team_per_min"] = diff / minutes
        out["direct_changes_in_long_gap"] = float(gap[cf].mean()) if len(cf) else np.nan
        # cambios directos por minuto dentro y fuera de los huecos largos
        out["direct_per_min_visible"] = float((~gap[cf]).sum() / (np.sum(mask & ~gap) / FPS / 60)) if len(cf) else 0.0
        out["direct_per_min_gap"] = (float(gap[cf].sum() / (np.sum(mask & gap) / FPS / 60))
                                     if len(cf) and (mask & gap).any() else 0.0)
        if proba is not None:
            pmax = proba.max(1)
            for label, sel in (("visible", mask & known), ("gap", mask & gap)):
                if sel.any():
                    out["pmax_p50_" + label] = float(np.median(pmax[sel]))
                    out["unsure_share_" + label] = float(np.mean(pmax[sel] < 0.7))
    return out


def concat_stats(items) -> dict:
    """Media ponderada por minutos de varios partidos (cada item = dict de shape_stats)."""
    total = sum(i["minutes"] for i in items)
    keys = [k for k in items[0] if k != "minutes"]
    return dict(minutes=total, **{k: float(np.nansum([i[k] * i["minutes"] for i in items if k in i]) /
                                         sum(i["minutes"] for i in items if k in i and np.isfinite(i[k])))
                                  for k in keys})


def bench_stats(oof: str, controlled: float, controlled_dfl: float):
    blob = np.load(os.path.join(_REPO_ROOT, "external_data", oof))
    gt_items, pred_items = [], []
    for i, name in enumerate(BENCH_GAMES_V3):
        match, raw = load_bench_game(name)
        t = controlled_dfl if is_dfl(name) else controlled
        gt = controlled_possession(raw, int(round(t * match.fps)))
        runs = pm.valid_runs(gt)
        known = ~np.isnan(degrade(match, BENCH_DEGRADE, seed=i).ball_x)
        proba = blob["p%d" % i]
        pred = decode(proba, runs, DecodeParams())
        gt_items.append(shape_stats(gt, runs, known))
        pred_items.append(shape_stats(pred, runs, known, proba))
    return concat_stats(gt_items), concat_stats(pred_items)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--team-name", default="La MasIA")
    parser.add_argument("--proba", default=os.path.join("data", "derived", "proba_l3.npz"))
    parser.add_argument("--controlled", type=float, default=1.5)
    parser.add_argument("--controlled-dfl", type=float, default=2.0)
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "docs", "investigacion", "resultados",
                                                      "real_vs_bench.json"))
    args = parser.parse_args(argv)

    pred_real = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", args.team_name + ".csv")))
    known_real, proba_real = None, None
    if os.path.exists(args.proba):
        blob = np.load(args.proba)
        known_real, proba_real = blob["ball_known"], blob["proba"]
    graded = [tuple(r) for r in GRADED]
    cols = {"reto: prediccion": shape_stats(pred_real, graded, known_real, proba_real)}
    gt_pub = pm.label_codes(pm.read_label_csv(GT_PATH))
    cols["publica: GT 5 min"] = shape_stats(gt_pub, [WINDOW])
    cols["publica: prediccion"] = shape_stats(pred_real, [WINDOW])
    cols["banco: GT"], cols["banco: prediccion"] = bench_stats(args.oof, args.controlled, args.controlled_dfl)

    keys = list(dict.fromkeys(k for c in cols.values() for k in c))
    print("{:<30}".format("medida") + "".join("{:>22}".format(c) for c in cols))
    for k in keys:
        print("{:<30}".format(k) + "".join("{:>22}".format("%.3f" % c[k] if k in c and np.isfinite(c[k]) else "-")
                                           for c in cols.values()))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(cols, handle, indent=1, ensure_ascii=False)
    print("resumen en", os.path.relpath(args.out, _REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
