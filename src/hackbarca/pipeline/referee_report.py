"""U8 (E094): arbitro y asistentes desde la pasada de video. Solo agregados y pseudoetiquetas (ADR-0019).

    scripts/run_aggregate.sh referee uv run python -m hackbarca.pipeline.referee_report

Las personas lejos de los dos colores de equipo (`video_conductor.video_people`, 20 % en E074) son
arbitros o porteros. Portero: dentro de un area; asistente: a < 2,5 m de la banda; arbitro: el resto, en
el campo. Por frame de la rejilla (1 de cada 5): velocidad del arbitro (la persona arbitro mas cercana a
la del frame anterior de la rejilla, < 3 m) y si hay arbitro visible.

Puerta (pseudoetiquetas: la entrega con pmax >= 0,9): AUC de -velocidad del arbitro para D frente a vivo,
junto a la misma AUC con la velocidad media de los jugadores del ATD (lo que el modelo ya tiene). Pasa si
la AUC del arbitro es >= 0,75 y la combinacion (media de rangos) sube >= 0,02 sobre los jugadores solos.
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.io.video_conductor import video_people
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.gate_shape import auc

EVERY = 5
DT = EVERY / 25


def rank(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    out[ok] = np.argsort(np.argsort(x[ok])) / max(ok.sum() - 1, 1)
    return out


def main() -> int:
    from hackbarca.io.load_match import load_match
    match = load_match()
    n, length, width = match.n_frames, match.pitch_length, match.pitch_width
    people, fr, _, far = video_people(match)
    x, y = people[:, 1], people[:, 2]
    box = ((x < 16.5) | (x > length - 16.5)) & (np.abs(y - width / 2) < 20.16)
    line = (y < 2.5) | (y > width - 2.5)
    ref = far & ~box & ~line & (x > 0) & (x < length) & (y > 0) & (y < width)
    assistants = np.bincount(fr[far & line], minlength=n)
    ref_x, ref_y = np.full(n, np.nan), np.full(n, np.nan)
    speed = np.full(n, np.nan)
    order = np.argsort(fr[ref], kind="stable")
    rf, rx, ry = fr[ref][order], x[ref][order], y[ref][order]
    starts = np.searchsorted(rf, np.unique(rf))
    for f, s, e in zip(np.unique(rf), starts, np.append(starts[1:], len(rf))):
        prev = f - EVERY
        if prev >= 0 and np.isfinite(ref_x[prev]):
            d = np.hypot(rx[s:e] - ref_x[prev], ry[s:e] - ref_y[prev])
            k = int(np.argmin(d))
            if d[k] < 3:
                speed[f] = d[k] / DT
        else:
            k = 0
        ref_x[f], ref_y[f] = rx[s + k], ry[s + k]

    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    pmax = np.load(os.path.join("data", "derived", "proba_l3.npz"))["proba"].max(1)
    grid = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        grid[a:b] = True
    grid &= np.arange(n) % EVERY == 0
    vx = np.gradient(match.players_x, axis=0) * 25
    vy = np.gradient(match.players_y, axis=0) * 25
    with np.errstate(all="ignore"):
        players = np.nanmean(np.hypot(vx, vy), axis=1)
    sure = grid & (pmax >= 0.9) & np.isin(pred, (0, 1, 2))
    known = ~np.isnan(match.ball_x)
    print("arbitro visible en %.3f de la rejilla; con velocidad %.3f; asistentes por frame %.2f" % (
        np.isfinite(ref_x[grid]).mean(), np.isfinite(speed[grid]).mean(), assistants[grid].mean()))
    for name, m in (("todo", sure), ("balon visible", sure & known), ("hueco del ATD", sure & ~known)):
        dead = pred[m] == 2
        a_ref, a_pl = auc(-speed[m], dead), auc(-players[m], dead)
        both = np.nanmean(np.stack([rank(-speed[m]), rank(-players[m])]), axis=0)
        print("%-14s D frente a vivo: AUC arbitro %.3f (cobertura %.3f) | jugadores ATD %.3f | juntos %.3f (n=%d)" % (
            name, a_ref, np.isfinite(speed[m]).mean(), a_pl, auc(both, dead), m.sum()))
    short = np.zeros(n, dtype=bool)
    for r in GRADED:
        segs = pm.segments(pred, [tuple(r)])
        for i, (a, b, c) in enumerate(segs):
            if c == 2 and b - a < 125 and 0 < i < len(segs) - 1:
                short[a:b] = True
    m = grid & short
    print("en los D cortos que quedan: velocidad del arbitro p50 %.2f m/s (vivo seguro %.2f, D largo seguro %.2f)" % (
        np.nanmedian(speed[m]), np.nanmedian(speed[sure & (pred != 2)]), np.nanmedian(speed[sure & (pred == 2) & ~short])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
