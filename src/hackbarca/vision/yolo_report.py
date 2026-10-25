"""Puerta C1: ¿sirve el balon del YOLO como evidencia de quien tiene el balon en los huecos?

    scripts/run_aggregate.sh yolo_report uv run python -m hackbarca.vision.yolo_report

ADR-0016 rechazo usar el YOLO afinado para *rellenar* el balon (en los parones el balon visto esta
en manos de quien saca y el modelo lo leia como posesion). Aqui se mide otro uso: como evidencia
blanda del equipo en juego vivo dentro de los huecos del ATD, donde el reto casi no tiene cambios
directos predichos (0,16 por minuto frente a 0,59 en el banco, E041).

Sobre el partido real (sin GT) y nuestra prediccion de la entrega:
1. cobertura del relleno en los huecos segun lo que predecimos (D frente a vivo) y segun la
   seguridad del modelo;
2. acuerdo entre "equipo del jugador mas cercano al balon del video" (a < 1,5 m) y el equipo que
   predecimos cuando el modelo esta seguro (pmax >= 0,9): mide si la senal es fiable;
3. dentro de los tramos vivos predichos de un solo equipo, cuantas veces el balon del video pasa
   >= 1 s pegado al otro equipo: candidatos a robos que no vemos, por minuto de hueco.
Solo agregados.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.io.load_match import AWAY, HOME, load_match
from hackbarca.io.video_fill import apply_video_fill
from hackbarca.pipeline.atd_stats import GRADED

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, A, D = 0, 1, 2
NEAR = 1.5      # m
SUSTAIN = 25    # frames


def nearest_team(match, x, y):
    """(equipo del mas cercano, su distancia) por frame; NaN/-1 sin balon."""
    dist = {}
    for team in (HOME, AWAY):
        cols = match.team_of_track == team
        with np.errstate(invalid="ignore"):
            d = np.hypot(match.players_x[:, cols] - x[:, None], match.players_y[:, cols] - y[:, None])
        dist[team] = np.where(np.isnan(d).all(1), np.inf, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))
    team = np.where(dist[HOME] <= dist[AWAY], H, A)
    best = np.minimum(dist[HOME], dist[AWAY])
    team[np.isnan(x) | ~np.isfinite(best)] = -1
    return team, best


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="La MasIA")
    parser.add_argument("--video-ball", default=os.path.join("data", "derived", "video_ball.npz"))
    parser.add_argument("--proba", default=os.path.join("data", "derived", "proba_l3.npz"))
    args = parser.parse_args(argv)

    match = load_match()
    filled = apply_video_fill(match, args.video_ball)
    pred = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", args.team_name + ".csv")))
    pmax = np.load(args.proba)["proba"].max(1)
    graded = np.zeros(match.n_frames, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    gap = np.isnan(match.ball_x) & graded
    has = gap & ~np.isnan(filled.ball_x)
    team, dist = nearest_team(match, filled.ball_x, filled.ball_y)
    live = (pred == H) | (pred == A)

    print("1. Cobertura del balon del video en los huecos del ATD (frames de hueco: %d)" % gap.sum())
    for label, sel in (("predicho D", gap & (pred == D)), ("predicho vivo", gap & live),
                       ("vivo y seguro (pmax>=0,9)", gap & live & (pmax >= 0.9)),
                       ("vivo y dudoso (pmax<0,7)", gap & live & (pmax < 0.7))):
        n = sel.sum()
        print("   {:<28} frames {:7d}  con balon de video {:5.1%}  de ellos a <{:.1f} m de alguien {:5.1%}".format(
            label, n, has[sel].mean() if n else np.nan, NEAR, (dist[sel & has] < NEAR).mean() if (sel & has).any() else np.nan))

    print("\n2. Acuerdo 'equipo mas cercano al balon del video' con el equipo predicho (juego vivo en hueco)")
    near = has & live & (dist < NEAR)
    for label, sel in (("pmax >= 0,9", near & (pmax >= 0.9)), ("0,7 <= pmax < 0,9", near & (pmax >= 0.7) & (pmax < 0.9)),
                       ("pmax < 0,7", near & (pmax < 0.7))):
        n = sel.sum()
        print("   {:<20} frames {:7d}  acuerdo {:.3f}".format(label, n, np.mean(team[sel] == pred[sel]) if n else np.nan))
    # referencia: el mismo acuerdo con el balon del propio ATD (donde lo ve), con el modelo seguro
    t_atd, d_atd = nearest_team(match, match.ball_x, match.ball_y)
    ref = graded & live & ~np.isnan(match.ball_x) & (d_atd < NEAR) & (pmax >= 0.9)
    print("   referencia: balon del ATD, pmax >= 0,9: frames %d  acuerdo %.3f" % (ref.sum(), np.mean(t_atd[ref] == pred[ref])))

    print("\n3. Candidatos a robo no visto: >= 1 s seguido del balon del video pegado al otro equipo")
    cands = 0
    gap_live_min = (gap & live).sum() / 25 / 60
    for a, b in runs_of(gap & live):
        own = pred[a]
        if (pred[a:b] != own).any():
            continue
        other = (team[a:b] == (1 - own)) & (dist[a:b] < NEAR) & has[a:b]
        cands += sum(1 for s, e in runs_of(other) if e - s >= SUSTAIN)
    print("   minutos de juego vivo predicho en hueco %.1f | candidatos %d (%.2f por minuto de hueco vivo)" % (
        gap_live_min, cands, cands / max(gap_live_min, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
