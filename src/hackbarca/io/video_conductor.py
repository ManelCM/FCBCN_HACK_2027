"""U6 (E075): quien conduce el balon segun el video tactico, real en el reto y simulado en el banco.

En el reto (vision/video_pass.py + video_report.py, solo agregados): 1 de cada 5 frames puntuados, la
persona detectada mas cercana (< 1,5 m) al candidato de balon mas fiable, con su equipo por el color
de la camiseta (0,926 de acierto frente al ATD); los colores lejanos (arbitros, porteros) se
descartan. Frente al equipo predicho por la entrega (E070), acuerdo 0,883 en frames seguros
(cobertura 0,52), 0,644 en dudosos (0,40) y 0,780 en huecos del ATD (0,57). Quitando el acierto del
modelo en cada contexto (medido en el banco, errores independientes) el video acierta ~0,90 en
seguros, ~0,87 en huecos y ~0,95 en dudosos: no depende del contexto y en los dudosos supera al modelo.

En el banco no hay video: se simula con el tracking limpio (el jugador mas cercano al balon de verdad
a < 1,5 m), con la misma rejilla de 5 frames, rachas de visibilidad (el detector pierde el balon a
tramos) y rachas de error (en los duelos elige al rival), calibradas para que cobertura y acierto
casen con lo medido en el reto.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

FPS = 25
H, A = 0, 1


@dataclass(frozen=True)
class VideoConductor:
    every: int = 5            # rejilla de la pasada de video (PERSON_EVERY)
    radius: float = 1.5       # m entre el balon y la persona
    keep: float = 0.94        # cuota del tiempo con el detector viendo balon y persona (calibrada, E075)
    vis_run_s: float = 1.5    # duracion media de una racha visible o no visible
    err: float = 0.03         # equipo equivocado anadido: la regla sola acierta 0,911 frente a la GT; el reto, ~0,90 (E075)
    err_run_s: float = 0.6    # duracion media de una racha de error
    dead_keep: float = 0.32   # con el balon muerto, cuota con conductor (el que va a sacar; reto, D seguro)
    dead_next: float = 0.70   # ... y cuota en la que es del equipo de la posesion siguiente (reto: 0,704)
    dead_run_s: float = 2.0   # rachas de conductor durante un paron
    change_err: float = 0.0   # prueba de estres (E075s): error extra cerca de un cambio H<->A (duelos)
    change_s: float = 1.5     # ... a menos de estos segundos del cambio


VIDEO_CONDUCTOR = VideoConductor()
# E075s: pesimista frente a lo medido en el reto: menos cobertura (~0,44 en vivo), el doble de errores
# sueltos, un 35 % de error cerca de cada robo y el lanzador del equipo siguiente solo el 60 %
VIDEO_CONDUCTOR_STRESS = VideoConductor(keep=0.80, err=0.06, change_err=0.35, dead_next=0.60)


def _nearest_team(px, py, team, bx, by, radius):
    """Equipo del jugador mas cercano al balon si esta a < radius; NaN si no."""
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    near = d[np.arange(len(j)), j] < radius
    return np.where(near, team[j].astype(float), np.nan)


def _runs(rng, n: int, share: float, mean_frames: float) -> np.ndarray:
    """Mascara booleana de rachas alternas (geometricas) con cuota `share` en True."""
    if share <= 0 or share >= 1:
        return np.full(n, share >= 1)
    out = np.zeros(n, dtype=bool)
    # las rachas duran >= 1 frame: si la de una cuota pequena sale corta, se alarga la otra para mantener la cuota
    on_len = mean_frames * 2 * share
    off_len = mean_frames * 2 * (1 - share)
    if on_len < 1:
        on_len, off_len = 1.0, (1 - share) / share
    elif off_len < 1:
        on_len, off_len = share / (1 - share), 1.0
    t, state = 0, bool(rng.random() < share)
    while t < n:
        mean = on_len if state else off_len
        length = int(rng.geometric(1.0 / max(mean, 1.0)))
        out[t:t + length] = state
        t += length
        state = not state
    return out


def simulate_conductor(match, gt: np.ndarray, sim: VideoConductor = VIDEO_CONDUCTOR, seed: int = 0) -> np.ndarray:
    """Equipo del conductor segun el video simulado (0/1, NaN sin observacion) por frame del banco limpio.

    En juego, el jugador mas cercano al balon; en los parones (GT D), con rachas, el que va a sacar: del
    equipo de la posesion siguiente con probabilidad dead_next.
    """
    rng = np.random.default_rng(seed + 4111)
    n = match.n_frames
    grid = np.arange(0, n, sim.every)
    team = _nearest_team(match.players_x[grid], match.players_y[grid], match.team_of_track,
                         match.ball_x[grid], match.ball_y[grid], sim.radius)
    seen = _runs(rng, len(grid), sim.keep, sim.vis_run_s * FPS / sim.every)
    wrong = _runs(rng, len(grid), sim.err, sim.err_run_s * FPS / sim.every)
    if sim.change_err > 0:
        # los errores del video se juntan en los duelos: cerca de cada cambio de equipo de la GT
        live = np.flatnonzero(np.isin(gt, (H, A)))
        chg = live[1:][gt[live[1:]] != gt[live[:-1]]]
        if len(chg):
            k = np.searchsorted(chg, grid)
            d = np.minimum(np.abs(grid - chg[np.maximum(k - 1, 0)]), np.abs(chg[np.minimum(k, len(chg) - 1)] - grid))
            wrong |= (d <= sim.change_s * FPS) & (rng.random(len(grid)) < sim.change_err)
    team = np.where(seen, team, np.nan)
    team = np.where(wrong & np.isfinite(team), 1 - team, team)
    dead = gt[grid] == 2
    idx = np.arange(n)
    nxt = np.where(np.isin(gt, (H, A)), idx, n)
    next_i = np.minimum.accumulate(nxt[::-1])[::-1][grid]
    next_team = np.where(next_i < n, gt[np.minimum(next_i, n - 1)], np.nan).astype(float)
    taker = _runs(rng, len(grid), sim.dead_keep, sim.dead_run_s * FPS / sim.every)
    other = _runs(rng, len(grid), 1 - sim.dead_next, sim.dead_run_s * FPS / sim.every)
    dead_team = np.where(other, 1 - next_team, next_team)
    team = np.where(dead, np.where(taker, dead_team, np.nan), team)
    out = np.full(n, np.nan)
    out[grid] = team
    return out


def _data(*parts):
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), *parts)


def video_people(match):
    """Personas de la pasada de video con su equipo por color: (people, fr, team_color, far).

    Centroides de color de cada equipo con las personas emparejadas a un slot del ATD (< 1,5 m; sin
    etiquetas: el equipo es el del slot); far = lejos de los dos colores (arbitros, porteros).
    """
    files = sorted(glob.glob(_data("derived", "video_pass_full", "block*.npz")))
    people = np.concatenate([np.load(f)["people"] for f in files])
    fr = people[:, 0].astype(int)
    d = np.hypot(match.players_x[fr] - people[:, 1:2], match.players_y[fr] - people[:, 2:3])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    team = match.team_of_track[j].astype(int)
    matched = d[np.arange(len(d)), j] < 1.5
    color = people[:, 4:7]
    # colores de cada equipo con las personas emparejadas al ATD (sin etiquetas: el equipo es el del slot)
    cents = np.stack([np.nanmedian(color[matched & (team == t)], axis=0) for t in (H, A)])
    dc = np.stack([np.linalg.norm(color - c, axis=1) for c in cents], axis=1)
    team_color = np.argmin(dc, axis=1)
    spread = np.median(np.linalg.norm(color[matched] - cents[team[matched]], axis=1))
    far = np.min(dc, axis=1) > 3 * spread
    return people, fr, team_color, far


def real_conductor(match) -> np.ndarray:
    """Equipo del conductor en el reto, de data/derived/video_pass_full/ (mismo criterio que video_report)."""
    files = sorted(glob.glob(_data("derived", "video_pass_full", "block*.npz")))
    ball = np.concatenate([np.load(f)["ball"] for f in files])
    people, fr, team_color, far = video_people(match)
    best = {}
    for f, x, y, c in ball:
        f = int(f)
        if f not in best or c > best[f][2]:
            best[f] = (x, y, c)
    out = np.full(match.n_frames, np.nan)
    order = np.argsort(fr, kind="stable")
    starts = np.searchsorted(fr[order], np.unique(fr))
    for f, s, e in zip(np.unique(fr), starts, np.append(starts[1:], len(fr))):
        if f not in best:
            continue
        idx = order[s:e]
        bx, by, _ = best[f]
        dd = np.hypot(people[idx, 1] - bx, people[idx, 2] - by)
        k = int(np.argmin(dd))
        if dd[k] < 1.5 and not far[idx[k]]:
            out[f] = team_color[idx[k]]
    return out


def _window_sum(x: np.ndarray, half: int) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(len(x))
    a, b = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    return cs[b] - cs[a]


def conductor_features(team: np.ndarray, every: int = 5) -> pd.DataFrame:
    """Familia "video_conductor": cuota de observaciones de cada equipo en ventanas centradas y su cambio."""
    obs = np.isfinite(team).astype(float)
    is_h = (team == H).astype(float)
    is_a = (team == A).astype(float)
    f = {}
    for half in (12, 50, 125):
        n = _window_sum(obs, half)
        slots = (2 * half + 1) / every
        with np.errstate(invalid="ignore", divide="ignore"):
            f["vc_m%d_h" % half] = np.where(n > 0, _window_sum(is_h, half) / n, np.nan)
            f["vc_m%d_a" % half] = np.where(n > 0, _window_sum(is_a, half) / n, np.nan)
        f["vc_n_m%d" % half] = n / slots
    for col in ("h", "a"):
        base = f["vc_m50_%s" % col]
        before = np.full(len(team), np.nan)
        after = np.full(len(team), np.nan)
        before[75:] = base[:-75]
        after[:-75] = base[75:]
        f["vc_chg75_%s" % col] = after - before
    return pd.DataFrame(f).astype(np.float32)


def calibrate() -> int:
    """Cobertura y acierto de la simulacion en el banco frente a lo medido en el reto (E075)."""
    from hackbarca.io.degrade import degrade
    from hackbarca.io.external import load_bench_game
    from hackbarca.pipeline.bench_config import BENCH_DEGRADE_V3_1, BENCH_GAMES_V3
    print("reto: cobertura 0,52 en vivo seguro, 0,57 en huecos, 0,32 en D seguro; acierto implicito ~0,90 (0,87 en huecos)")
    for label, sim in (("base", VIDEO_CONDUCTOR), ("estres", VIDEO_CONDUCTOR_STRESS)):
        _calibrate(label, sim, degrade, load_bench_game, BENCH_DEGRADE_V3_1, BENCH_GAMES_V3)
    return 0


def _calibrate(label, sim, degrade, load_bench_game, BENCH_DEGRADE_V3_1, BENCH_GAMES_V3):
    cov_live, cov_gap, cov_dead, hit = [], [], [], []
    for i, name in enumerate(BENCH_GAMES_V3):
        match, gt = load_bench_game(name)
        team = simulate_conductor(match, gt, sim, seed=i)
        known = ~np.isnan(degrade(match, BENCH_DEGRADE_V3_1, seed=i).ball_x)
        grid = np.zeros(match.n_frames, dtype=bool)
        grid[::VIDEO_CONDUCTOR.every] = True
        live = grid & np.isin(gt, (H, A))
        cov_live.append(np.isfinite(team[live & known]))
        cov_gap.append(np.isfinite(team[live & ~known]))
        cov_dead.append(np.isfinite(team[grid & (gt == 2)]))
        ok = live & np.isfinite(team)
        hit.append(team[ok] == gt[ok])
    print("banco %s: cobertura %.3f en vivo con balon visible, %.3f en huecos, %.3f en D; acierto frente a la GT %.3f" % (label,
        np.mean(np.concatenate(cov_live)), np.mean(np.concatenate(cov_gap)), np.mean(np.concatenate(cov_dead)),
        np.mean(np.concatenate(hit))))


def shift_report() -> int:
    """Desajuste reto/banco de las columnas vc_* (solo percentiles; ADR-0004).

        scripts/run_aggregate.sh vc_shift uv run python -m hackbarca.io.video_conductor --shift
    """
    from hackbarca.io.external import load_bench_game
    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED
    from hackbarca.pipeline.bench_config import BENCH_GAMES_V3
    from hackbarca.pipeline.feature_shift import shift_table
    match = load_match()
    graded = np.zeros(match.n_frames, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    real = conductor_features(real_conductor(match))[graded]
    bench = pd.concat([conductor_features(simulate_conductor(*load_bench_game(n), seed=i))
                       for i, n in enumerate(BENCH_GAMES_V3)], ignore_index=True)
    for col, pr, pb, gap in sorted(shift_table(real, bench, list(real.columns)), key=lambda r: -r[3]):
        print("{:<16} reto {:>18} banco {:>18} desaj. {:.2f}".format(
            col, "/".join("%.2f" % v for v in pr), "/".join("%.2f" % v for v in pb), gap))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(shift_report() if "--shift" in sys.argv else calibrate())
