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


SLOW_STEP = 0.05   # m por frame (1,25 m/s): balon "lento" (E043)


def _run_length_per_frame(mask: np.ndarray) -> np.ndarray:
    """Por frame, la longitud de la racha True a la que pertenece (0 fuera)."""
    out = np.zeros(len(mask), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = b - a
    return out


def ball_dynamics(ball_x: np.ndarray, ball_y: np.ndarray, graded: np.ndarray, live: np.ndarray | None = None,
                  fps: int = 25) -> dict:
    """Agregados de la dinamica del balon: objetivos para calibrar el simulador (E013, E047).

    Solo necesita el balon, asi que sirve igual para el ATD real que para el banco degradado sin
    jugadores. `live` (opcional) marca el juego vivo para medir el balon lento en vivo (E043).
    """
    from hackbarca.features.frame_features import TOUCH_DV, TOUCH_W, _centered_max
    n = len(ball_x)
    missing = np.isnan(ball_x)
    seen = graded & ~missing
    out = {"ball_missing": float(missing[graded].mean())}
    # residuo frente a la media movil de 9 frames (suavidad del tracker), como en describe()
    kernel = np.ones(9) / 9
    resid = np.where(missing, np.nan, ball_x) - np.convolve(np.nan_to_num(ball_x), kernel, mode="same")
    full = np.convolve((~missing).astype(float), kernel, mode="same") > 0.999
    ok = seen & full & ~np.isnan(resid)
    out["resid_std"] = float(np.std(resid[ok])) if ok.any() else np.nan
    # cambio de velocidad antes/despues (el touch_dv de frame_features), huecos <= 10 interpolados
    idx = np.arange(n)
    bx, by = ball_x.copy(), ball_y.copy()
    if (~missing).sum() >= 2:
        fill = missing & (_run_length_per_frame(missing) <= 10)
        bx[fill] = np.interp(idx[fill], idx[~missing], ball_x[~missing])
        by[fill] = np.interp(idx[fill], idx[~missing], ball_y[~missing])
    w = TOUCH_W
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[w:, 0], before[w:, 1] = (bx[w:] - bx[:-w]) * fps / w, (by[w:] - by[:-w]) * fps / w
    after[:-w, 0], after[:-w, 1] = (bx[w:] - bx[:-w]) * fps / w, (by[w:] - by[:-w]) * fps / w
    dv = np.hypot(after[:, 0] - before[:, 0], after[:, 1] - before[:, 1])
    dvs = dv[seen & np.isfinite(dv)]
    for q in (50, 90, 99):
        out["dv_p%d" % q] = float(np.percentile(dvs, q)) if len(dvs) else np.nan
    dvz = np.nan_to_num(dv)
    touch = (dvz > TOUCH_DV) & (dvz >= _centered_max(dvz, 5))
    out["touches_per_s"] = float(touch[seen].sum() / max(seen.sum() / fps, 1e-9))
    speed = np.full(n, np.nan)
    speed[4:] = np.hypot(ball_x[4:] - ball_x[:-4], ball_y[4:] - ball_y[:-4]) * fps / 4
    sp = speed[seen & np.isfinite(speed)]
    for q in (50, 90, 99):
        out["speed_p%d" % q] = float(np.percentile(sp, q)) if len(sp) else np.nan
    starts = np.flatnonzero(np.diff(missing.astype(np.int8)) == 1)    # ultimo frame con balon
    starts = starts[graded[starts] & (starts >= 4)]
    out["speed_before_gap_p50"] = float(np.nanpercentile(speed[starts], 50)) if len(starts) else np.nan
    if live is not None:
        step = np.full(n, np.nan)
        step[1:] = np.hypot(np.diff(ball_x), np.diff(ball_y))
        vis = graded & live & np.isfinite(step)
        slow = vis & (step < SLOW_STEP)
        out["live_slow_share"] = float(slow.sum() / max(vis.sum(), 1))
        out["live_slow_runs_per_min"] = float((_runs(slow) >= fps).sum() / max(vis.sum() / fps / 60, 1e-9))
    return out


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
    from hackbarca.features.frame_features import TOUCH_DV, _centered_max, frame_features
    dv = np.nan_to_num(frame_features(match, version=2)["touch_dv"].to_numpy())
    touch = (dv > TOUCH_DV) & (dv >= _centered_max(dv, 5))
    seen = graded & ~ball_missing
    print("toques por segundo con balon visible (banco ~0,49): {:.2f}  dv mediano={:.1f} m/s".format(
        touch[seen].sum() / max(seen.sum() / match.fps, 1e-9), float(np.median(dv[seen]))))
    raw_dv = frame_features(match, version=2)["touch_dv"].to_numpy()
    print("dv balon (m/s) p50/p90/p99/p99.9: " + " / ".join(
        "{:.1f}".format(v) for v in np.nanpercentile(raw_dv[seen], [50, 90, 99, 99.9])))
    speed = np.full(match.n_frames, np.nan)
    speed[4:] = np.hypot(match.ball_x[4:] - match.ball_x[:-4], match.ball_y[4:] - match.ball_y[:-4]) * match.fps / 4
    print("velocidad balon (m/s) p50/p90/p99: " + " / ".join(
        "{:.1f}".format(v) for v in np.nanpercentile(speed[seen], [50, 90, 99])))
    starts = np.flatnonzero(np.diff(ball_missing.astype(np.int8)) == 1)  # ultimo frame con balon
    starts = starts[graded[starts] & (starts >= 4)]
    print("velocidad balon justo antes de perderlo (m/s) p50/p90: " + " / ".join(
        "{:.1f}".format(v) for v in np.nanpercentile(speed[starts], [50, 90])) if len(starts) else "sin huecos")
    # suavidad: autocorrelacion del ruido de posicion (residuo frente a media movil de 9 frames)
    k = 9
    kernel = np.ones(k) / k
    bx = np.where(ball_missing, np.nan, match.ball_x)
    resid = bx - np.convolve(np.nan_to_num(bx), kernel, mode="same")
    full = np.convolve((~ball_missing).astype(float), kernel, mode="same") > 0.999
    ok = seen & full & ~np.isnan(resid)
    print("residuo de posicion del balon frente a media movil de 9 frames: std={:.2f} m".format(
        float(np.std(resid[ok]))))
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
    parser.add_argument("--json", default=None,
                        help="solo la dinamica del balon del reto (objetivos de calibracion, E047) a este json")
    args = parser.parse_args(argv)

    if args.json:
        # vivo = lo que predice la entrega (sin GT): cota inferior del balon lento en vivo, porque
        # parte de los D cortos que predecimos es vivo (E041-E043); ventana publica con su GT
        import json
        from hackbarca.io.load_match import load_match
        match = load_match()
        graded = np.zeros(match.n_frames, dtype=bool)
        for start, stop in GRADED:
            graded[start:stop] = True
        pred = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", "La MasIA.csv")))
        public = load_gt_codes(PUBLIC_GT)
        out = {"reto (vivo predicho)": ball_dynamics(match.ball_x, match.ball_y, graded, (pred == pm.H) | (pred == pm.A)),
               "ventana publica (vivo GT)": ball_dynamics(match.ball_x, match.ball_y, public != pm.X,
                                                         (public == pm.H) | (public == pm.A))}
        for name, row in out.items():
            print(name + ": " + ", ".join("%s=%.3f" % kv for kv in row.items()))
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1, ensure_ascii=False)
        return 0

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
