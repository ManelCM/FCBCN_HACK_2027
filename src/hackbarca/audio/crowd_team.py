"""A5 (E095): la grada dice quien ataca? Solo agregados y pseudoetiquetas (ADR-0019; Q6 sin responder).

    scripts/run_aggregate.sh crowd_team uv run python -m hackbarca.audio.crowd_team

Hipotesis: en casa, la grada se anima cuando ataca el local y se calla (o pita) cuando ataca el
visitante. Grada = Cheering + Crowd + Applause de PANNs (`audioset_gate`, ya calculado), media en 3 s,
y su cambio frente a los 3 s anteriores. Pseudoetiquetas: la entrega con pmax >= 0,9 en juego vivo.

Puerta (la comun de T3-T4): AUC >= 0,75 para "posee el local", en todo el juego vivo seguro y con el
balon en el ultimo tercio, con cobertura >= 10 % de los frames dudosos (pmax < 0,7) y de los huecos.
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.audio.audioset_gate import to_tactical_series
from hackbarca.audio.whistle import half_offsets
from hackbarca.pipeline.gate_shape import auc

WIN = 75   # 3 s


def _mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(x))
    a, b = np.clip(idx + lo, 0, len(x)), np.clip(idx + hi + 1, 0, len(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cn[b] - cn[a] > 0, (cs[b] - cs[a]) / (cn[b] - cn[a]), np.nan)


def main() -> int:
    from hackbarca.eval.evaluate import pm
    from hackbarca.io.load_match import load_match
    from hackbarca.pipeline.atd_stats import GRADED

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    blob = np.load(os.path.join(data_dir, "derived", "audioset.npz"))
    series, names = blob["series"], [str(n) for n in blob["names"]]
    alignment = np.load(os.path.join(data_dir, "derived", "broadcast_alignment.npz"))
    match = load_match()
    n = match.n_frames
    tac = to_tactical_series(series, half_offsets(alignment["tactical_to_broadcast"], GRADED), n)
    crowd = np.nansum(tac[:, [names.index(c) for c in ("Cheering", "Crowd", "Applause")]], axis=1)
    crowd[~np.isfinite(tac[:, 0])] = np.nan
    level = _mean(crowd, -WIN // 2, WIN // 2)
    change = level - _mean(crowd, -WIN - WIN // 2, -WIN // 2)

    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    pmax = np.load(os.path.join(data_dir, "derived", "proba_l3.npz"))["proba"].max(1)
    graded = np.zeros(n, dtype=bool)
    third = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
        home_left = np.nanmean(match.players_x[a:b][:, match.team_of_track == 0]) < match.pitch_length / 2
        bx = match.ball_x[a:b]
        # ultimo tercio del que tiene el balon: el local ataca hacia la derecha si empieza a la izquierda
        attack_h = bx > 2 * match.pitch_length / 3 if home_left else bx < match.pitch_length / 3
        attack_a = bx < match.pitch_length / 3 if home_left else bx > 2 * match.pitch_length / 3
        third[a:b] = np.where(pred[a:b] == 0, attack_h, attack_a)
    sure = graded & (pmax >= 0.9) & np.isin(pred, (0, 1))
    print("cobertura del audio en juego vivo seguro: %.3f" % np.isfinite(level[sure]).mean())
    for name, m in (("vivo seguro", sure), ("ultimo tercio", sure & third)):
        home = pred[m] == 0
        print("%-14s posee el local: AUC grada %.3f | cambio de grada %.3f (n=%d)" % (
            name, auc(level[m], home), auc(change[m], home), m.sum()))
    doubt = graded & (pmax < 0.7)
    gap = graded & np.isnan(match.ball_x) & np.isin(pred, (0, 1))
    print("cobertura: dudosos %.3f, huecos %.3f" % (np.isfinite(level[doubt]).mean(), np.isfinite(level[gap]).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
