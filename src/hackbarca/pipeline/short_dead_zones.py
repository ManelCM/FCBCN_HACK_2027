"""E090b: donde caen los D cortos (< 5 s entre dos tramos vivos), en el reto y en el banco. Solo cuotas.

Si los D cortos de mas del reto fueran saques de banda o de esquina rapidos (D de verdad), la regla de E090
haria dano; si caen en el area con el mismo equipo a los dos lados (portero con el balon en las manos), son
juego vivo mal leido. Zona por la posicion media del balon en el tramo:
area (16,5 m x 40,3 m), banda (< 3 m de la linea de banda), esquina (< 4 m del banderin), centro (< 4 m del
punto central), resto, sin balon (el ATD no lo ve en todo el tramo).

    scripts/run_aggregate.sh short_dead_zones uv run python -m hackbarca.pipeline.short_dead_zones --oof <oof v3.1>
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np

from hackbarca.eval.bench import pm
from hackbarca.io.degrade import degrade
from hackbarca.io.external import load_bench_game
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.bench_config import BENCH_DEGRADE_V3_1, BENCH_GAMES_V3

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, A, D = 0, 1, 2
SHORT = 125
ZONES = ("area", "banda", "esquina", "centro", "resto", "sin balon")


def zone(x, y, length, width):
    if not np.isfinite(x):
        return "sin balon"
    if min(np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)) < 4:
        return "esquina"
    if np.hypot(x - length / 2, y - width / 2) < 4:
        return "centro"
    if (x < 16.5 or x > length - 16.5) and abs(y - width / 2) < 20.16:
        return "area"
    if y < 3 or y > width - 3:
        return "banda"
    return "resto"


def dead_runs(codes, runs, bx, by, length, width):
    """Counter de (clase, zona, mismo equipo a los dos lados) de los tramos D entre dos tramos vivos."""
    out = Counter()
    for s, e in runs:
        segs = pm.segments(codes, [(s, e)])
        for n, (a, b, code) in enumerate(segs):
            if code != D or not (0 < n < len(segs) - 1):
                continue
            if segs[n - 1][2] not in (H, A) or segs[n + 1][2] not in (H, A):
                continue
            with np.errstate(all="ignore"):
                z = zone(np.nanmean(bx[a:b]), np.nanmean(by[a:b]), length, width)
            out[("corto" if b - a < SHORT else "largo", z, segs[n - 1][2] == segs[n + 1][2])] += 1
    return out


def show(name, c, minutes):
    for cls in ("corto", "largo"):
        tot = sum(v for k, v in c.items() if k[0] == cls)
        if not tot:
            continue
        parts = []
        for z in ZONES:
            n = sum(v for k, v in c.items() if k[0] == cls and k[1] == z)
            same = c[(cls, z, True)]
            if n:
                parts.append("%s %.2f (mismo eq. %.2f)" % (z, n / tot, same / n))
        print("%-28s D %-5s %.3f/min | %s" % (name, cls, tot / minutes, " | ".join(parts)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="fichero en external_data/")
    parser.add_argument("--team-name", default="La MasIA")
    args = parser.parse_args(argv)
    blob = np.load(os.path.join(_REPO_ROOT, "external_data", args.oof))
    gt_c, pr_c, minutes = Counter(), Counter(), 0.0
    for i, name in enumerate(BENCH_GAMES_V3):
        match, _ = load_bench_game(name)
        gt = blob["gt%d" % i].astype(np.int8)
        runs = pm.valid_runs(gt)
        pred = decode(blob["p%d" % i], runs, DecodeParams())
        seen = degrade(match, BENCH_DEGRADE_V3_1, seed=i)
        gt_c += dead_runs(gt, runs, match.ball_x, match.ball_y, match.pitch_length, match.pitch_width)
        pr_c += dead_runs(pred, runs, seen.ball_x, seen.ball_y, match.pitch_length, match.pitch_width)
        minutes += sum(e - s for s, e in runs) / 25 / 60
    show("banco: GT (balon limpio)", gt_c, minutes)
    show("banco: prediccion", pr_c, minutes)

    from hackbarca.io.load_match import load_match
    match = load_match()
    codes = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", args.team_name + ".csv")))
    graded = [tuple(r) for r in GRADED]
    c = dead_runs(codes, graded, match.ball_x, match.ball_y, match.pitch_length, match.pitch_width)
    show("reto: prediccion", c, sum(b - a for a, b in graded) / 25 / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
