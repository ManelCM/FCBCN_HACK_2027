"""U10 (E097): puerta de las acciones con el balon del broadcast (T-DEED). Solo agregados (ADR-0004, ADR-0019).

    scripts/run_aggregate.sh ball_action_gate uv run python -m hackbarca.pipeline.ball_action_gate

Lee data/derived/ball_actions/ (`vision.ball_action_pass`), pasa cada paso del broadcast a frame tactico
con el desfase de su parte y saca eventos: pico de la probabilidad de alguna accion >= umbral, a >= 1 s del
anterior. El equipo del evento es el lado (izquierda/derecha del video) con mas probabilidad. Lado ->
local/visitante: un bit por parte, el que mas acuerda con la prediccion segura (se imprimen los dos para ver
que cambia en el descanso). Pseudoetiquetas: la entrega con pmax >= 0,9.

Puerta (la comun de T3-T4):
  - equipo: acuerdo de las acciones con balon (pase, conduccion, cabezazo, pase alto, centro, tiro) con la
    prediccion segura en juego vivo >= 0,85, con cobertura >= 10 % de los frames dudosos o de los huecos;
  - D: fuera, saque de banda y falta con un D predicho a <= 3 s, lift >= 1,5 sobre instantes al azar.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.pipeline.atd_stats import GRADED

FPS = 25
CLASSES = ("PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
           "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL")
WITH_BALL = [CLASSES.index(c) for c in ("PASS", "DRIVE", "HEADER", "HIGH PASS", "CROSS", "SHOT")]
STOPS = [CLASSES.index(c) for c in ("OUT", "THROW IN", "FREE KICK")]
GAP = 25        # frames tacticos entre eventos (1 s)
NEAR = 25       # cobertura: evento a <= 1 s


def _data(*parts):
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), *parts)


def load_steps(offsets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame tactico, parte, proba (T, 25)) de todos los pasos que caen en una parte."""
    frames, halves, probas = [], [], []
    for f in sorted(glob.glob(_data("derived", "ball_actions", "half*.npz"))):
        blob = np.load(f)
        b, p = blob["frame"], blob["proba"].astype(np.float32)
        for k, (lo, hi, offset) in enumerate(offsets):
            m = (b >= lo) & (b < hi)
            frames.append(b[m] + offset)
            halves.append(np.full(m.sum(), k))
            probas.append(p[m])
    return np.concatenate(frames), np.concatenate(halves), np.concatenate(probas)


def events(frames: np.ndarray, proba: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices de paso de los eventos, su clase y su lado (1 = derecha)."""
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


def near(ev: np.ndarray, t: np.ndarray, before: int, after: int) -> np.ndarray:
    """Para cada t, hay un evento en [t - before, t + after]."""
    if not len(ev):
        return np.zeros(len(t), dtype=bool)
    lo = np.searchsorted(ev, t - before)
    hi = np.searchsorted(ev, t + after, side="right")
    return hi > lo


def main() -> int:
    from hackbarca.audio.whistle import half_offsets
    from hackbarca.io.load_match import load_match
    match = load_match()
    n = match.n_frames
    alignment = np.load(_data("derived", "broadcast_alignment.npz"))
    offsets = half_offsets(alignment["tactical_to_broadcast"], GRADED)
    frames, half, proba = load_steps(offsets)
    order = np.argsort(frames, kind="stable")
    frames, half, proba = frames[order], half[order], proba[order]
    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    pmax = np.load(_data("derived", "proba_l3.npz"))["proba"].max(1)
    graded = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    minutes = graded.sum() / FPS / 60
    doubt = np.flatnonzero(graded & (pmax < 0.7))
    gaps = np.flatnonzero(graded & np.isnan(match.ball_x))
    print("pasos del modelo en partes puntuadas: %d (%.1f por s de juego puntuado)" % (len(frames), len(frames) / (minutes * 60)))

    for thr in (0.2, 0.3, 0.5):
        keep, cls, side = events(frames, proba, thr)
        t = frames[keep]
        ok = graded[t]
        keep, cls, side, t, h = keep[ok], cls[ok], side[ok], t[ok], half[keep][ok]
        print("\numbral %.1f: %.1f eventos por minuto | %s" % (thr, len(t) / minutes, " ".join(
            "%s:%.2f" % (c.split()[0][:5].lower() + (c.split()[1][:2].lower() if " " in c else ""),
                         (cls == k).sum() / minutes) for k, c in enumerate(CLASSES))))
        ball = np.isin(cls, WITH_BALL)
        sure = (pmax[t] >= 0.9) & np.isin(pred[t], (0, 1))
        team = np.full(len(t), -1)
        for k in range(len(offsets)):
            m = ball & sure & (h == k)
            # lado derecha = local (0) o visitante (1): el que mas acuerda, por parte
            agree_right_home = ((side[m] == 1) == (pred[t[m]] == 0)).mean() if m.any() else np.nan
            right_home = agree_right_home >= 0.5
            sel = h == k
            team[sel] = np.where(side[sel] == 1, 0 if right_home else 1, 1 if right_home else 0)
            print("  parte %d: acuerdo si derecha = local %.3f, si derecha = visitante %.3f (n=%d)" % (
                k + 1, agree_right_home, 1 - agree_right_home, m.sum()))
        m = ball & sure
        print("  equipo de las acciones con balon frente a la prediccion segura: %.3f (n=%d)" % (
            (team[m] == pred[t[m]]).mean() if m.any() else np.nan, m.sum()))
        for name in ("PASS", "DRIVE", "SHOT"):
            mm = m & (cls == CLASSES.index(name))
            if mm.any():
                print("    %-6s %.3f (n=%d)" % (name.lower(), (team[mm] == pred[t[mm]]).mean(), mm.sum()))
        tb = np.sort(t[ball])
        print("  cobertura (accion con balon a <= 1 s): dudosos %.3f, huecos del ATD %.3f" % (
            near(tb, doubt, NEAR, NEAR).mean(), near(tb, gaps, NEAR, NEAR).mean()))
        md = ball & (pmax[t] < 0.7) & np.isin(pred[t], (0, 1))
        print("  en dudosos, equipo de la accion = prediccion final %.3f (n=%d)" % (
            (team[md] == pred[t[md]]).mean() if md.any() else np.nan, md.sum()))
        # D: tras fuera, saque de banda y falta
        dead = np.flatnonzero(graded & (pred == 2))
        stop = np.isin(cls, STOPS)
        rng = np.random.default_rng(0)
        base_t = rng.choice(np.flatnonzero(graded), 20000, replace=False)
        hit = near(dead, t[stop], 3 * FPS, 3 * FPS).mean() if stop.any() else np.nan
        base = near(dead, base_t, 3 * FPS, 3 * FPS).mean()
        print("  paradas (fuera, banda, falta): %.2f por minuto | con D predicho a <= 3 s %.3f, base %.3f, lift %.2f" % (
            stop.sum() / minutes, hit, base, hit / base))
        print("  goles detectados: %d" % (cls == CLASSES.index("GOAL")).sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
