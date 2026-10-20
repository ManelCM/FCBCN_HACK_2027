"""Estadisticas AGREGADAS del ATD real para calibrar el simulador de degradacion (ADR-0006).

ADR-0004: lo ejecuta una persona. Solo imprime porcentajes, percentiles y metricas; ninguna fila,
coordenada ni etiqueta por frame. La salida se puede pegar en el chat con el agente.

    uv run python -m hackbarca.pipeline.atd_stats
    uv run python -m hackbarca.pipeline.atd_stats --bench   # lo mismo sobre el banco degradado
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hackbarca.eval.evaluate import load_gt_codes, pm
from hackbarca.models.baseline import NOBODY, nearest_team, predict_baseline

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PUBLIC_GT = os.path.join(_REPO_ROOT, "external", "more-than-a-hack-2027", "evaluation", "gt",
                         "BAR-ATH_possession_gt_public.csv")
# Ventanas evaluadas del partido completo (CHALLENGE.md / DATA.md), [start, stop)
GRADED = ((11185, 90652), (110808, 190198))


def _runs(mask: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def _pct(values, qs=(10, 50, 90)) -> str:
    if len(values) == 0:
        return "n=0"
    return "n={} ".format(len(values)) + " ".join(
        "p{}={:.0f}".format(q, np.percentile(values, q)) for q in qs)


def describe(match, gt: np.ndarray | None, graded: np.ndarray, title: str) -> None:
    ball_missing = np.isnan(match.ball_x)
    visible = ~np.isnan(match.players_x)
    print("==", title)
    print("frames={} graded={} slots={} (home={}, away={})".format(
        match.n_frames, int(graded.sum()), match.n_slots,
        int((match.team_of_track == 0).sum()), int((match.team_of_track == 1).sum())))
    print("balon ausente: total={:.3f} graded={:.3f}".format(
        ball_missing.mean(), ball_missing[graded].mean()))
    print("huecos de balon en graded (frames): " + _pct(_runs(ball_missing & graded)))
    per_frame = visible.sum(1)[graded]
    print("tracks visibles/frame en graded: " + _pct(per_frame, (5, 50, 95)))
    for side, name in ((0, "home"), (1, "away")):
        cols = match.team_of_track == side
        print("  {} visibles/frame: {}".format(name, _pct(visible[:, cols].sum(1)[graded], (5, 50, 95))))
    lengths = np.concatenate([_runs(visible[:, s]) for s in range(match.n_slots)])
    print("longitud de tracklets (frames): " + _pct(lengths, (10, 50, 90)))
    print("speed jugadores no-NaN={:.3f} speed balon no-NaN={:.3f}".format(
        np.mean(~np.isnan(match.players_speed)[visible]) if visible.any() else 0.0,
        np.mean(~np.isnan(match.ball_speed)[~ball_missing]) if (~ball_missing).any() else 0.0))

    if gt is not None:
        labelled = gt != pm.X
        alive = labelled & ((gt == pm.H) | (gt == pm.A))
        dead = labelled & (gt == pm.D)
        print("con GT ({} frames): balon ausente | vivo={:.3f}  | muerto={:.3f}".format(
            int(labelled.sum()), ball_missing[alive].mean(), ball_missing[dead].mean()))
        near = nearest_team(match)
        has = alive & (near != NOBODY)
        truth = np.where(gt == pm.H, 0, 1)
        print("acierto del mas cercano en vivo con balon: {:.3f}".format(
            np.mean(near[has] == truth[has]) if has.any() else float("nan")))
        for w in (0, 25, 50, 125):
            r = pm.evaluate(gt, predict_baseline(match, w))
            print("baseline w={:<3d} spellF1={:.4f} macroF1={:.4f} poss={} (gt {})".format(
                w, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="calcular sobre el banco externo degradado en vez de sobre el ATD")
    args = parser.parse_args(argv)

    if args.bench:
        from hackbarca.io.degrade import degrade
        from hackbarca.io.external import load_metrica_bench
        for i, game in enumerate(("1", "2", "3")):
            match, gt = load_metrica_bench(game)
            describe(degrade(match, seed=i), gt, gt != pm.X, "metrica game {} degradado".format(game))
        return 0

    from hackbarca.io.load_match import load_match
    match = load_match()
    graded = np.zeros(match.n_frames, dtype=bool)
    for start, stop in GRADED:
        graded[start:stop] = True
    public = load_gt_codes(PUBLIC_GT)
    describe(match, None, graded, "ATD BAR-ATH partido completo (sin GT)")
    window = public != pm.X
    describe(match, public, window, "ATD BAR-ATH ventana publica 5 min (GT publica)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
