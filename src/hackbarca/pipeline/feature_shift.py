"""Desajuste de dominio de las features: ATD real frente al banco degradado (solo agregados).

La leccion de E012-E013: unas features pueden funcionar en el banco y no significar lo mismo en el
ATD real (el balon real va filtrado). Antes de aceptar una version de features se comparan sus
percentiles en el partido del reto y en el banco.

ADR-0004: lee data/, asi que solo imprime percentiles por feature:

    scripts/run_aggregate.sh feature_shift uv run python -m hackbarca.pipeline.feature_shift --features 6
"""

from __future__ import annotations

import argparse

import numpy as np

from hackbarca.io.degrade import degrade
from hackbarca.io.external import load_bench_game
from hackbarca.models.l3 import features_for
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES_V3

PCTS = (10, 50, 90)


def _graded_mask(n: int) -> np.ndarray:
    mask = np.zeros(n, bool)
    for a, b in GRADED:
        mask[a:b] = True
    return mask


def shift_table(real, bench, columns) -> list[tuple]:
    """(feature, percentiles reales, percentiles banco, desajuste) con el desajuste medido como la
    diferencia de medianas en unidades del rango intercuartilico del banco."""
    rows = []
    for col in columns:
        r, b = real[col].to_numpy(float), bench[col].to_numpy(float)
        r, b = r[np.isfinite(r)], b[np.isfinite(b)]
        if not len(r) or not len(b):
            continue
        pr, pb = np.percentile(r, PCTS), np.percentile(b, PCTS)
        iqr = np.subtract(*np.percentile(b, (75, 25))) or 1e-9
        rows.append((col, pr, pb, abs(pr[1] - pb[1]) / iqr))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--only-new", action="store_true", help="solo las columnas que no estan en v3")
    args = parser.parse_args(argv)

    import pandas as pd

    from hackbarca.io.load_match import load_match
    halftime = (GRADED[0][1] + GRADED[1][0]) // 2
    match = load_match()
    real = features_for(match, None, halftime, args.features)
    real = real[_graded_mask(len(real))]
    bench = []
    for i, game in enumerate(BENCH_GAMES_V3):
        m, gt = load_bench_game(game)
        df = features_for(degrade(m, BENCH_DEGRADE, seed=i), None, version=args.features)
        bench.append(df[gt != 3])
    bench = pd.concat(bench, ignore_index=True)
    columns = list(real.columns)
    if args.only_new:
        base = set(features_for(degrade(*[load_bench_game(BENCH_GAMES_V3[0])[0]], BENCH_DEGRADE, seed=0),
                                None, version=3).columns)
        columns = [c for c in columns if c not in base]
    rows = shift_table(real, bench, columns)
    print("{:<26} {:>24} {:>24} {:>8}".format("feature", "reto p10/p50/p90", "banco p10/p50/p90", "desaj."))
    for col, pr, pb, gap in sorted(rows, key=lambda r: -r[3]):
        print("{:<26} {:>24} {:>24} {:>8.2f}".format(
            col, "/".join("%.2f" % v for v in pr), "/".join("%.2f" % v for v in pb), gap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
