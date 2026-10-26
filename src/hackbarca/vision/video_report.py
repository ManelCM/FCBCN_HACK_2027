"""Puertas de la pasada de video (U3, U5, U5b, U6, U8; E074-E075). Solo agregados (ADR-0004).

    scripts/run_aggregate.sh video_report uv run python -m hackbarca.vision.video_report

Lee data/derived/video_pass_full/ (vision/video_pass.py) y mide:
  U5  equipo por color: cada persona detectada se empareja con el jugador del ATD mas cercano a < 1,5 m
      (cuyo equipo conocemos); con los frames pares se aprende el color medio de cada equipo (centroides
      en Lab) y con los impares se mide el acierto del vecino mas cercano. Las personas lejos de los dos
      colores son arbitros o porteros (U8, recuento).
  U5b completar el ATD: personas a > 3 m de cualquier jugador del ATD, por frame, donde el ATD ve < 20.
  U6  quien conduce: persona mas cercana al candidato de balon mas fiable (< 1,5 m) y su equipo por color,
      frente al equipo predicho (seguro / dudoso / hueco).
  U3  toques: trayectoria del balon del video (candidato mas fiable enlazado a < 3 m del anterior) y picos
      de aceleracion >= 4 m/s^2 junto a una persona (Link y Hoernig 2017); toques por minuto y AUC del
      "toque de una persona del otro equipo" frente a los cambios H<->A predichos.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.real_signals import _data

H, A, D = 0, 1, 2


def load_pass():
    files = sorted(glob.glob(os.path.join(_data("derived", "video_pass_full"), "block*.npz")))
    ball = np.concatenate([np.load(f)["ball"] for f in files])
    people = np.concatenate([np.load(f)["people"] for f in files])
    return ball, people


def main() -> int:
    from hackbarca.io.load_match import load_match
    match = load_match()
    ball, people = load_pass()
    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    pmax = np.load(_data("derived", "proba_l3.npz"))["proba"].max(1)
    n = match.n_frames
    print("pasada: %d candidatos de balon, %d personas en %d frames con personas" % (
        len(ball), len(people), len(np.unique(people[:, 0]))))

    # U5: emparejar personas con jugadores del ATD
    fr = people[:, 0].astype(int)
    px, py = match.players_x[fr], match.players_y[fr]
    d = np.hypot(px - people[:, 1:2], py - people[:, 2:3])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    dmin = d[np.arange(len(d)), j]
    team = match.team_of_track[j].astype(int)
    matched = dmin < 1.5
    color = people[:, 4:7]
    train = matched & (fr % 2 == 0)
    test = matched & (fr % 2 == 1)
    cents = np.stack([np.nanmedian(color[train & (team == t)], axis=0) for t in (H, A)])
    dc = np.stack([np.linalg.norm(color - c, axis=1) for c in cents], axis=1)
    team_color = np.argmin(dc, axis=1)
    acc = np.mean(team_color[test] == team[test])
    print("\nU5 equipo por color: emparejadas con el ATD %.1f%% | acierto en frames apartados %.3f (n=%d)" % (
        100 * matched.mean(), acc, test.sum()))
    spread = np.median(np.linalg.norm(color[train] - cents[team[train]], axis=1))
    far = np.min(dc, axis=1) > 3 * spread
    print("U8 personas lejos de los dos colores (arbitros/porteros): %.1f%% de las detectadas" % (100 * far.mean()))

    # U5b: completar el ATD
    visible = (~np.isnan(match.players_x)).sum(1)
    new = (dmin > 3) & ~far
    per_frame = np.bincount(fr[new], minlength=n)
    frames = np.unique(fr)
    low = frames[visible[frames] < 20]
    print("U5b frames con el ATD por debajo de 20 jugadores: %d de %d; personas nuevas por frame ahi: media %.2f, "
          ">= 2 en el %.1f%%" % (len(low), len(frames), per_frame[low].mean() if len(low) else np.nan,
                                 100 * np.mean(per_frame[low] >= 2) if len(low) else np.nan))

    # U6: quien conduce segun el video
    graded = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    best = {}
    for f, x, y, c in ball:
        f = int(f)
        if f not in best or c > best[f][2]:
            best[f] = (x, y, c)
    agree = {"seguro": [], "dudoso": [], "hueco": []}
    for f in frames:
        if f not in best or not graded[f] or pred[f] not in (H, A):
            continue
        sel = fr == f
        bxx, byy, _ = best[f]
        dd = np.hypot(people[sel, 1] - bxx, people[sel, 2] - byy)
        k = np.argmin(dd)
        if dd[k] > 1.5 or far[sel][k]:
            continue
        ok = team_color[sel][k] == pred[f]
        if pmax[f] >= 0.9:
            agree["seguro"].append(ok)
        elif pmax[f] < 0.7:
            agree["dudoso"].append(ok)
        if np.isnan(match.ball_x[f]):
            agree["hueco"].append(ok)
    for k, v in agree.items():
        print("U6 conductor por video vs equipo predicho (%s): acuerdo %.3f (n=%d)" % (k, np.mean(v) if v else np.nan, len(v)))
    graded_frames = np.flatnonzero(graded)
    cover = np.mean([f in best for f in graded_frames[::25]])
    print("U3 frames puntuados con algun candidato de balon: %.3f" % cover)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
