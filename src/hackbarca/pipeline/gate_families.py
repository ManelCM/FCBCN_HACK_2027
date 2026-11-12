"""Puerta de las familias de entrenador (T2: F3 E058, F5 E059, F6 E071), banco v3.1, datos externos.

    uv run python -m hackbarca.pipeline.gate_families

Calcula `features/coach.py` sobre los 10 partidos degradados con v3.1 (misma semilla que el banco):
  - senales de equipo (F3, F5): AUC de "posee el local" de local - visitante en juego vivo, en
    huecos largos y en los 0-3 s tras un cambio H<->A (como gate_shape/gate_control);
  - senales de estado (F6): AUC de D frente a vivo, en todo y dentro de los huecos largos (en
    los huecos el vivo sale D el 22-25 %, E040).
Puerta (plan completo): equipo, AUC tras robo >= 0,78 o en hueco >= 0,75; estado, AUC en hueco por
encima de la mejor referencia actual (velocidad colectiva). Despues, desajuste reto/banco < 0,25.
"""

from __future__ import annotations

import numpy as np

from hackbarca.io.degrade import degrade
from hackbarca.io.external import controlled_possession, is_dfl, load_bench_game
from hackbarca.pipeline.bench_config import BENCH_DEGRADE_V3_1, BENCH_GAMES_V3
from hackbarca.pipeline.gate_control import context_masks
from hackbarca.pipeline.gate_shape import auc

H, A, D = 0, 1, 2


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="transition+territory+gamestate",
                        help="familias de features/coach.py separadas por '+'")
    args = parser.parse_args(argv)
    names = tuple(args.families.split("+"))
    from hackbarca.features.coach import extra_features
    team_store, dead_store = {}, {}
    for i, name in enumerate(BENCH_GAMES_V3):
        match, raw = load_bench_game(name)
        gt = controlled_possession(raw, int(round((2.0 if is_dfl(name) else 1.5) * match.fps)))
        deg = degrade(match, BENCH_DEGRADE_V3_1, seed=i)
        cols = extra_features(deg, names)
        masks = context_masks(gt, np.isnan(deg.ball_x))
        live = masks["vivo"]
        dead = gt == D
        gap_all = masks["en hueco"] | (dead & np.isnan(deg.ball_x))
        for col in cols.columns:
            if col.endswith("_h") and col[:-2] + "_a" in cols:
                diff = (cols[col] - cols[col[:-2] + "_a"]).to_numpy()
                for ctx, m in masks.items():
                    team_store.setdefault(col[:-2], {}).setdefault(ctx, ([], []))
                    team_store[col[:-2]][ctx][0].append(diff[m])
                    team_store[col[:-2]][ctx][1].append(gt[m] == H)
            elif not col.endswith("_a"):
                values = cols[col].to_numpy()
                for ctx, m in (("todo", live | dead), ("en hueco", gap_all & (live | dead))):
                    dead_store.setdefault(col, {}).setdefault(ctx, ([], []))
                    dead_store[col][ctx][0].append(values[m])
                    dead_store[col][ctx][1].append(dead[m])
        print("listo", name, flush=True)

    if team_store:
        print()
        print("senales de equipo - AUC de 'posee el local' (|AUC - 0,5| es la fuerza)")
        print("{:<26}{:>10}{:>10}{:>16}".format("senal", "vivo", "en hueco", "0-3 s tras robo"))
        rows = []
        for sig, ctxs in team_store.items():
            vals = [auc(np.concatenate(ctxs[c][0]), np.concatenate(ctxs[c][1])) for c in ("vivo", "en hueco", "0-3 s tras robo")]
            rows.append((max(abs(vals[2] - 0.5), abs(vals[1] - 0.5)), sig, vals))
        for _, sig, vals in sorted(rows, reverse=True):
            print("{:<26}{:>10.3f}{:>10.3f}{:>16.3f}".format(sig, *vals))
    if dead_store:
        print()
        print("senales de estado - AUC de 'balon muerto' (|AUC - 0,5| es la fuerza)")
        print("{:<26}{:>10}{:>10}".format("senal", "todo", "en hueco"))
        rows = []
        for sig, ctxs in dead_store.items():
            vals = [auc(np.concatenate(ctxs[c][0]), np.concatenate(ctxs[c][1])) for c in ("todo", "en hueco")]
            rows.append((abs(vals[1] - 0.5), sig, vals))
        for _, sig, vals in sorted(rows, reverse=True)[:25]:
            print("{:<26}{:>10.3f}{:>10.3f}".format(sig, *vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
