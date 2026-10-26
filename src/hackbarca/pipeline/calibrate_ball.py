"""Calibracion del balon del simulador para el banco v3.1 (E047, ADR-0017).

    uv run python -m hackbarca.pipeline.calibrate_ball

E043: el error correlado del balon de v2.1/v3 (AR(1) sin suavizar) mete ~4 m/s de velocidad aparente
y borra el balon lento del juego vivo (tramos lentos >= 1 s por minuto: limpio 1,08, degradado 0,01,
reto 0,55 en el vivo predicho y 1,86 en la ventana publica con GT). Aqui se busca, solo con datos
externos, la combinacion de suavizado y error correlado que reproduce a la vez:
  - la dinamica del balon del ATD real (E013; objetivos de `atd_stats --json`): residuo, dv,
    velocidad, toques, velocidad antes de un hueco;
  - el balon lento en juego vivo, cerca del limpio (no se puede medir bien en el reto sin GT).
Solo se degrada el balon (`degrade._long_ball` con la misma semilla y el mismo orden de numeros
aleatorios que `degrade`), asi que cada punto de la rejilla cuesta segundos.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, replace

import numpy as np

from hackbarca.io.degrade import _long_ball, _nearest_distance
from hackbarca.io.external import controlled_possession, is_dfl, load_bench_game
from hackbarca.pipeline.atd_stats import ball_dynamics
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES_V3

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TARGETS = os.path.join(_REPO_ROOT, "docs", "investigacion", "resultados", "objetivos_balon_reto.json")
OUT = os.path.join(_REPO_ROOT, "docs", "investigacion", "resultados", "calibracion_balon_v3_1.json")
H, A, X = 0, 1, 3

# (clave de ball_dynamics, escala del error). El balon lento en vivo apunta al limpio del banco
# (1,08 tramos/min, 8,8 % de frames; E043): en el reto solo hay una cota inferior (0,55 con el vivo
# predicho) y la ventana publica (1,86) es demasiado corta para ser el objetivo.
SCALES = {"resid_std": 0.02, "dv_p50": 0.5, "dv_p90": 1.0, "dv_p99": 2.0, "speed_p50": 0.5,
          "speed_p90": 1.0, "speed_p99": 2.0, "speed_before_gap_p50": 0.5, "touches_per_s": 0.01,
          "live_slow_share": 0.02, "live_slow_runs_per_min": 0.25}
SLOW_FLOOR = 0.55     # nunca por debajo de lo que ya se ve en el reto
# 1.ª pasada (E047a): con ball_noise = 0,02 fijo, ninguna combinacion de suavizado del error correlado
# devolvia tramos lentos (0,01/min): el ruido blanco que se suma al final (~0,04 m de salto por frame)
# corta cualquier racha de 25 frames por debajo de 0,05 m. 2.ª pasada: ball_noise entra en la rejilla.
GRID = dict(ball_noise=(0.0, 0.005, 0.01, 0.02), ball_smooth_sigma=(6.0, 10.0), ball_corr_noise=(0.4, 0.8),
            ball_corr_tau=(75.0,), ball_corr_smooth=(0.0, 10.0, 25.0))


def targets() -> dict:
    with open(TARGETS, encoding="utf-8") as handle:
        real = json.load(handle)["reto (vivo predicho)"]
    out = {k: real[k] for k in SCALES if k in real}
    out["live_slow_share"], out["live_slow_runs_per_min"] = 0.088, 1.08   # limpio (E043)
    return out


def loss(stats: dict, goal: dict) -> float:
    total = sum(((stats[k] - goal[k]) / SCALES[k]) ** 2 for k in SCALES if k in goal and np.isfinite(stats[k]))
    if stats["live_slow_runs_per_min"] < SLOW_FLOOR:
        total += 100.0
    return float(total)


def bench_inputs():
    """Por partido: (match, visible, weight, graded, vivo) con la GT controlada del banco."""
    out = []
    for name in BENCH_GAMES_V3:
        match, raw = load_bench_game(name)
        gt = controlled_possession(raw, int(round((2.0 if is_dfl(name) else 1.5) * match.fps)))
        visible = ~np.isnan(match.ball_x)
        occluded = visible & (_nearest_distance(match) < BENCH_DEGRADE.occlusion_radius)
        weight = np.where(occluded, BENCH_DEGRADE.occlusion_boost, 1.0)
        out.append((match, visible, weight, gt != X, (gt == H) | (gt == A)))
    return out


def pooled_stats(params, inputs) -> dict:
    """Media ponderada por frames puntuados de ball_dynamics sobre los 10 partidos."""
    rows, weights = [], []
    for i, (match, visible, weight, graded, live) in enumerate(inputs):
        rng = np.random.default_rng(i)          # misma semilla que bench_feature_paths / degrade
        bx, by = _long_ball(rng, match, params, visible, weight)
        rows.append(ball_dynamics(bx, by, graded, live, match.fps))
        weights.append(graded.sum())
    w = np.array(weights, dtype=float)
    return {k: float(np.average([r[k] for r in rows], weights=w)) for k in rows[0]}


def main() -> int:
    goal = targets()
    inputs = bench_inputs()
    results = []
    print("objetivo: " + ", ".join("%s=%.3f" % kv for kv in goal.items()))
    ref = pooled_stats(BENCH_DEGRADE, inputs)
    print("v3 actual  loss=%.1f  " % loss(ref, goal) + ", ".join("%s=%.3f" % (k, ref[k]) for k in SCALES))
    for combo in itertools.product(*GRID.values()):
        params = replace(BENCH_DEGRADE, **dict(zip(GRID, combo)))
        stats = pooled_stats(params, inputs)
        results.append(dict(params=dict(zip(GRID, combo)), loss=loss(stats, goal), stats=stats))
    results.sort(key=lambda r: r["loss"])
    print("\nmejores 10:")
    for r in results[:10]:
        s = r["stats"]
        print("loss=%6.1f  %s | resid %.3f dv %.1f/%.1f/%.1f vel %.1f/%.1f antes-hueco %.1f toques %.3f "
              "lento %.3f tramos/min %.2f" % (r["loss"], r["params"], s["resid_std"], s["dv_p50"], s["dv_p90"],
                                               s["dv_p99"], s["speed_p50"], s["speed_p90"],
                                               s["speed_before_gap_p50"], s["touches_per_s"],
                                               s["live_slow_share"], s["live_slow_runs_per_min"]))
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(dict(goal=goal, v3=dict(params=asdict(BENCH_DEGRADE), loss=loss(ref, goal), stats=ref),
                       grid=results), handle, indent=1, ensure_ascii=False)
    print("guardado", os.path.relpath(OUT, _REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
