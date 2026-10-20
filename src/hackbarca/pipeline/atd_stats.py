"""Estadisticas AGREGADAS del ATD real para calibrar el simulador de degradacion (ADR-0006).

ADR-0004: lo ejecuta una persona. Solo imprime porcentajes, percentiles y metricas; ninguna fila,
coordenada ni etiqueta por frame. La salida se puede pegar en el chat con el agente.

    uv run python -m hackbarca.pipeline.atd_stats
    uv run python -m hackbarca.pipeline.atd_stats --bench   # lo mismo sobre el banco degradado
"""

from __future__ import annotations

import argparse
import os
import warnings

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
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "n=0"
    return "n={} ".format(len(values)) + " ".join(
        "p{}={:.0f}".format(q, np.percentile(values, q)) for q in qs)


LONG_GAP = 100


def collective_speed(match) -> np.ndarray:
    """Mediana por frame de la velocidad (m/s) de los jugadores, derivada de posiciones a 1 s."""
    lag = match.fps
    vx = match.players_x[lag:] - match.players_x[:-lag]
    vy = match.players_y[lag:] - match.players_y[:-lag]
    out = np.full(match.n_frames, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames sin ningun jugador
        out[lag // 2:lag // 2 + len(vx)] = np.nanmedian(np.sqrt(vx * vx + vy * vy), axis=1)
    return out


def _near_boundary(x, y, match, margin=3.0):
    return ((x < margin) | (x > match.pitch_length - margin) |
            (y < margin) | (y > match.pitch_width - margin))


def long_gap_report(match, graded: np.ndarray, windows=None, gt=None) -> None:
    """Son parones los huecos largos sin balon? Solo agregados."""
    missing = np.isnan(match.ball_x)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    gaps = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a > LONG_GAP and graded[a:b].any() and a > 0 and b < match.n_frames]
    in_long = np.zeros(match.n_frames, dtype=bool)
    for a, b in gaps:
        in_long[a:b] = True
    speed = collective_speed(match)
    visible_players = (~np.isnan(match.players_x)).sum(1)
    print("-- huecos largos (> {} frames): n={} cubren {:.3f} de lo evaluado".format(
        LONG_GAP, len(gaps), (in_long & graded).sum() / max(graded.sum(), 1)))
    if not gaps:
        return
    print("velocidad colectiva mediana (m/s): en hueco largo={:.2f}  con balon={:.2f}".format(
        np.nanmedian(speed[in_long & graded]), np.nanmedian(speed[~missing & graded])))
    print("jugadores visibles mediana: en hueco largo={:.0f}  con balon={:.0f}".format(
        np.median(visible_players[in_long & graded]), np.median(visible_players[~missing & graded])))
    before = np.array([_near_boundary(match.ball_x[a - 1], match.ball_y[a - 1], match) for a, _ in gaps])
    after = np.array([_near_boundary(match.ball_x[b], match.ball_y[b], match) for _, b in gaps])
    print("balon a < 3 m de una linea: justo antes del hueco={:.2f}  justo despues={:.2f}".format(
        before.mean(), after.mean()))
    # velocidad en los primeros/ultimos 2 s del hueco frente al centro: un paron se "calma"
    mid = [np.nanmedian(speed[a + (b - a) // 4:b - (b - a) // 4]) for a, b in gaps]
    print("velocidad colectiva mediana en el centro de cada hueco: " + _pct(np.array(mid) * 100, (10, 50, 90))
          + "  (cm/s)")
    if windows:
        starts = np.array([s for s, _ in windows])
        ends = np.array([b for _, b in gaps])
        tagged = np.mean([np.any((starts >= b - 75) & (starts <= b + 250)) for b in ends])
        print("huecos largos con un SET PIECE que empieza en [fin-3 s, fin+10 s]: {:.2f} ({} SET PIECES)".format(
            tagged, len(windows)))
    if gt is not None:
        alive = (gt == pm.H) | (gt == pm.A)
        dead = gt == pm.D
        print("con GT: velocidad colectiva mediana vivo={:.2f}  muerto={:.2f} (m/s)".format(
            np.nanmedian(speed[alive]), np.nanmedian(speed[dead])))
        if in_long[alive | dead].any():
            print("con GT: frames de hueco largo que son muerto={:.3f}".format(
                np.mean(dead[in_long & (alive | dead)])))


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
        from hackbarca.io.external import load_metrica_bench, metrica_events
        from hackbarca.io.smart_tagging import simulate_set_piece_windows
        from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES
        for i, game in enumerate(BENCH_GAMES):
            match, gt = load_metrica_bench(game)
            degraded = degrade(match, BENCH_DEGRADE, seed=i)
            windows = simulate_set_piece_windows(metrica_events(game), seed=i)
            describe(degraded, gt, gt != pm.X, "metrica game {} degradado".format(game))
            long_gap_report(degraded, gt != pm.X, windows, gt)
        return 0

    from hackbarca.io.load_match import load_match
    from hackbarca.io.smart_tagging import set_piece_windows
    match = load_match()
    windows = set_piece_windows()
    graded = np.zeros(match.n_frames, dtype=bool)
    for start, stop in GRADED:
        graded[start:stop] = True
    public = load_gt_codes(PUBLIC_GT)
    describe(match, None, graded, "ATD BAR-ATH partido completo (sin GT)")
    long_gap_report(match, graded, windows)
    window = public != pm.X
    describe(match, public, window, "ATD BAR-ATH ventana publica 5 min (GT publica)")
    long_gap_report(match, window, windows, np.where(window, public, pm.X))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
