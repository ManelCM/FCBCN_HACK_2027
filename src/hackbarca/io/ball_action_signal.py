"""U10 (E097-E098): equipo de las acciones con el balon del broadcast, real en el reto y simulado en el banco.

En el reto (vision/ball_action_pass.py, T-DEED de SoccerNet Team Ball 2025 sobre el broadcast): eventos
de accion con balon (pase, conduccion, cabezazo, pase alto, centro, tiro) con pico >= THR a >= 1 s del
anterior, pasados a frame tactico con el desfase de cada parte. El lado del video (izquierda/derecha) es
el del campo donde esta cada equipo en el ATD de esa parte, sin pseudoetiquetas (E098 comprueba que
coincide con el bit elegido en E097). Frente a la prediccion segura acuerda 0,886 con THR = 0,3 (E097).

En el banco no hay broadcast: se simulan eventos en el juego vivo de la GT con el ritmo medido en el reto
y el equipo de la GT, con errores segun la distancia al cambio de equipo mas cercano (los duelos), y
eventos sueltos durante los parones con el equipo de la posesion siguiente o el contrario.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

FPS = 25
H, A = 0, 1
THR = 0.3
GAP = 25        # frames entre eventos (1 s)
WITH_BALL = (0, 1, 2, 3, 5, 7)   # PASS, DRIVE, HEADER, HIGH PASS, CROSS, SHOT (orden de vision.ball_action_pass)


@dataclass(frozen=True)
class BallActions:
    # medido en el reto frente a la entrega (E098). Todo desacuerdo cuenta como error del video: prudente
    live_rate: float = 14.5   # eventos por minuto de juego vivo
    dead_rate: float = 1.1    # eventos por minuto de parada
    err_near: float = 0.33    # equipo equivocado a < 1,5 s de un cambio de equipo (acuerdo 0,667)
    err_mid: float = 0.14     # ... entre 1,5 y 5 s (0,856)
    err_far: float = 0.09     # ... a mas de 5 s (0,910)
    dead_next: float = 0.54   # en las paradas, cuota de eventos con el equipo de la posesion siguiente
    lite: bool = False        # E098b: sin la ventana de 1 s, su recuento ni ba_flip (ball_action_features)


BALL_ACTIONS = BallActions()
BALL_ACTIONS_LITE = BallActions(lite=True)


def _data(*parts):
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), *parts)


def _steps(offsets):
    """(frame tactico, parte, proba) de los pasos de la pasada que caen en una parte, ordenados."""
    frames, halves, probas = [], [], []
    for f in sorted(glob.glob(_data("derived", "ball_actions", "half*.npz"))):
        blob = np.load(f)
        b, p = blob["frame"], blob["proba"].astype(np.float32)
        for k, (lo, hi, offset) in enumerate(offsets):
            m = (b >= lo) & (b < hi)
            frames.append(b[m] + offset)
            halves.append(np.full(m.sum(), k))
            probas.append(p[m])
    frames, halves, probas = np.concatenate(frames), np.concatenate(halves), np.concatenate(probas)
    order = np.argsort(frames, kind="stable")
    return frames[order], halves[order], probas[order]


def pick_events(frames: np.ndarray, proba: np.ndarray, thr: float = THR):
    """Indices de paso de los eventos (pico >= thr a >= GAP frames del anterior), su clase y su lado (1 = derecha)."""
    left, right = proba[:, 1::2], proba[:, 2::2]
    action = left + right
    best = action.max(1)
    cand = np.flatnonzero((best >= thr) & (best >= np.roll(best, 1)) & (best >= np.roll(best, -1)))
    cand = cand[np.argsort(-best[cand], kind="stable")]
    taken = np.zeros(len(frames), dtype=bool)
    keep = []
    for i in cand:                                   # supresion de no maximos a 1 s
        lo, hi = np.searchsorted(frames, [frames[i] - GAP, frames[i] + GAP])
        if not taken[lo:hi].any():
            keep.append(i)
            taken[i] = True
    keep = np.sort(np.array(keep, dtype=int))
    cls = action[keep].argmax(1)
    side = (right[keep, cls] > left[keep, cls]).astype(int)
    return keep, cls, side


def right_team(match, offsets) -> list[int]:
    """Por parte, el equipo que esta a la derecha del campo en el ATD (x media de sus jugadores > largo / 2)."""
    out = []
    for lo, hi, offset in offsets:
        a, b = lo + offset, hi + offset
        home_x = np.nanmean(match.players_x[a:b][:, match.team_of_track == H])
        out.append(A if home_x < match.pitch_length / 2 else H)
    return out


def real_ball_actions(match, thr: float = THR, flip: bool = False) -> np.ndarray:
    """Equipo (0/1) de cada accion con balon en su frame tactico; NaN en el resto. flip: lado contrario."""
    from hackbarca.audio.whistle import half_offsets
    from hackbarca.pipeline.atd_stats import GRADED
    alignment = np.load(_data("derived", "broadcast_alignment.npz"))
    offsets = half_offsets(alignment["tactical_to_broadcast"], GRADED)
    frames, half, proba = _steps(offsets)
    keep, cls, side = pick_events(frames, proba, thr)
    ball = np.isin(cls, WITH_BALL)
    keep, side = keep[ball], side[ball]
    rights = np.array(right_team(match, offsets))
    if flip:
        rights = 1 - rights
    r = rights[half[keep]]
    team = np.where(side == 1, r, 1 - r)
    out = np.full(match.n_frames, np.nan)
    t = frames[keep]
    ok = (t >= 0) & (t < match.n_frames)
    out[t[ok]] = team[ok]
    return out


def simulate_ball_actions(gt: np.ndarray, sim: BallActions = BALL_ACTIONS, seed: int = 0) -> np.ndarray:
    """Equipo de las acciones con balon simuladas (0/1 en el frame del evento, NaN en el resto)."""
    rng = np.random.default_rng(seed + 7919)
    n = len(gt)
    live = np.isin(gt, (H, A))
    dead = gt == 2
    p = np.where(live, sim.live_rate, np.where(dead, sim.dead_rate, 0.0)) / (60 * FPS)
    cand = np.flatnonzero(rng.random(n) < p)
    keep, last = [], -GAP
    for t in cand:                                   # >= 1 s entre eventos, como en el reto
        if t - last >= GAP:
            keep.append(t)
            last = t
    t = np.array(keep, dtype=int)
    out = np.full(n, np.nan)
    if not len(t):
        return out
    team = gt[t].astype(float)
    # distancia al cambio de equipo mas cercano de la GT (solo juego vivo)
    lv = np.flatnonzero(live)
    chg = lv[1:][gt[lv[1:]] != gt[lv[:-1]]]
    if len(chg):
        k = np.searchsorted(chg, t)
        d = np.minimum(np.abs(t - chg[np.maximum(k - 1, 0)]), np.abs(chg[np.minimum(k, len(chg) - 1)] - t))
    else:
        d = np.full(len(t), np.inf)
    err = np.where(d < 1.5 * FPS, sim.err_near, np.where(d < 5 * FPS, sim.err_mid, sim.err_far))
    wrong = rng.random(len(t)) < err
    team = np.where(wrong, 1 - team, team)
    # paradas: equipo de la posesion siguiente (o el contrario)
    idx = np.arange(n)
    nxt = np.where(live, idx, n)
    next_i = np.minimum.accumulate(nxt[::-1])[::-1][t]
    next_team = np.where(next_i < n, gt[np.minimum(next_i, n - 1)], H).astype(float)
    in_dead = dead[t]
    other = rng.random(len(t)) >= sim.dead_next
    team = np.where(in_dead, np.where(other, 1 - next_team, next_team), team)
    out[t] = team
    return out


def _window_sum(x: np.ndarray, half: int) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(len(x))
    a, b = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    return cs[b] - cs[a]


def ball_action_features(team: np.ndarray, lite: bool = False) -> pd.DataFrame:
    """Familia "ball_actions": cuota de acciones de cada equipo en ventanas, y la ultima y la siguiente accion.

    lite (E098b): quita lo que mas pesa a < 1,5 s de un cambio, donde el equipo acierta 0,667 (E098)."""
    n = len(team)
    obs = np.isfinite(team).astype(float)
    is_h = (team == H).astype(float)
    is_a = (team == A).astype(float)
    f = {}   # siempre en pares _h/_a: swap_teams los intercambia al aumentar los datos
    for half in (75, 150) if lite else (25, 75, 150):
        k = _window_sum(obs, half)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["ba_m%d_h" % half] = np.where(k > 0, _window_sum(is_h, half) / k, np.nan)
            f["ba_m%d_a" % half] = np.where(k > 0, _window_sum(is_a, half) / k, np.nan)
        if half == 25:   # los recuentos de ventanas largas no casan: en el reto llegan en rachas (E098, desajuste 1,0)
            f["ba_n_m%d" % half] = k
    idx = np.arange(n)
    seen = np.where(obs > 0, idx, -1)
    last = np.maximum.accumulate(seen)
    nxt = np.where(obs > 0, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    has_last, has_next = last >= 0, nxt < n
    last_team = np.where(has_last, team[np.maximum(last, 0)], np.nan)
    next_team = np.where(has_next, team[np.minimum(nxt, n - 1)], np.nan)
    f["ba_last_h"] = last_team == H
    f["ba_last_s"] = np.where(has_last, np.minimum(idx - last, 750) / FPS, np.nan)
    f["ba_next_h"] = next_team == H
    f["ba_next_s"] = np.where(has_next, np.minimum(nxt - idx, 750) / FPS, np.nan)
    f["ba_last_h"] = np.where(has_last, f["ba_last_h"], np.nan)
    f["ba_next_h"] = np.where(has_next, f["ba_next_h"], np.nan)
    f["ba_last_a"] = np.where(has_last, last_team == A, np.nan)
    f["ba_next_a"] = np.where(has_next, next_team == A, np.nan)
    if lite:
        return pd.DataFrame(f).astype(np.float32)
    with np.errstate(invalid="ignore"):
        f["ba_flip"] = np.where(has_last & has_next, (last_team != next_team).astype(float), np.nan)
    return pd.DataFrame(f).astype(np.float32)


def calibrate_real() -> int:
    """E098: lo que la simulacion necesita, medido en el reto con la entrega (solo agregados, ADR-0004).

        scripts/run_aggregate.sh ba_calib uv run python -m hackbarca.io.ball_action_signal
    """
    from hackbarca.eval.evaluate import pm
    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED
    match = load_match()
    n = match.n_frames
    team = real_ball_actions(match)
    flipped = real_ball_actions(match, flip=True)
    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    pmax = np.load(_data("derived", "proba_l3.npz"))["proba"].max(1)
    graded = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    ev = graded & np.isfinite(team)
    sure = graded & (pmax >= 0.9)
    live_sure = sure & np.isin(pred, (H, A))
    dead_sure = sure & (pred == 2)
    print("lado por la geometria del ATD: acuerdo con la prediccion segura %.3f (lado contrario %.3f)" % (
        (team[ev & live_sure] == pred[ev & live_sure]).mean(), (flipped[ev & live_sure] == pred[ev & live_sure]).mean()))
    for name, m in (("vivo seguro", live_sure), ("D seguro", dead_sure),
                    ("vivo (todo)", graded & np.isin(pred, (H, A))), ("D (todo)", graded & (pred == 2))):
        print("eventos por minuto en %-12s %.2f" % (name, (ev & m).sum() / (m.sum() / FPS / 60)))
    segs = [s for r in GRADED for s in pm.segments(pred, [tuple(r)])]
    chg = np.array([b0 for (a0, b0, c0), (a1, b1, c1) in zip(segs[:-1], segs[1:]) if {c0, c1} == {H, A} and b0 == a1])
    t = np.flatnonzero(ev & live_sure)
    k = np.searchsorted(chg, t)
    d = np.minimum(np.abs(t - chg[np.maximum(k - 1, 0)]), np.abs(chg[np.minimum(k, len(chg) - 1)] - t))
    for name, m in (("< 1,5 s", d < 1.5 * FPS), ("1,5-5 s", (d >= 1.5 * FPS) & (d < 5 * FPS)), ("> 5 s", d >= 5 * FPS)):
        print("acuerdo en vivo seguro a %-8s de un cambio predicho: %.3f (n=%d)" % (name, (team[t[m]] == pred[t[m]]).mean(), m.sum()))
    t = np.flatnonzero(ev & dead_sure)
    idx = np.arange(n)
    nxt = np.where(np.isin(pred, (H, A)), idx, n)
    next_i = np.minimum.accumulate(nxt[::-1])[::-1][t]
    ok = next_i < n
    print("en D seguro, evento con el equipo de la posesion siguiente: %.3f (n=%d)" % (
        (team[t[ok]] == pred[next_i[ok]]).mean(), ok.sum()))
    return 0


def shift_report() -> int:
    """Desajuste reto/banco de las columnas ba_* (solo percentiles; ADR-0004).

        scripts/run_aggregate.sh ba_shift uv run python -m hackbarca.io.ball_action_signal --shift
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
    real = ball_action_features(real_ball_actions(match))[graded]
    bench = pd.concat([ball_action_features(simulate_ball_actions(load_bench_game(n)[1], seed=i))
                       for i, n in enumerate(BENCH_GAMES_V3)], ignore_index=True)
    for col, pr, pb, gap in sorted(shift_table(real, bench, list(real.columns)), key=lambda r: -r[3]):
        print("{:<12} reto {:>24} banco {:>24} desaj. {:.2f}".format(
            col, "/".join("%.2f" % v for v in pr), "/".join("%.2f" % v for v in pb), gap))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(shift_report() if "--shift" in sys.argv else calibrate_real())
