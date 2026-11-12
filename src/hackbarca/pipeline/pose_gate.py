"""U7 (E091): puerta de la pose (saque de banda, portero con el balon). Solo agregados (ADR-0004, ADR-0019).

    scripts/run_aggregate.sh pose_gate uv run python -m hackbarca.pipeline.pose_gate

Lee la pasada de pose (`vision.pose_pass`, data/derived/pose_pass/) y construye, en la rejilla de 1 de cada 5
frames puntuados, dos senales:
  - saque de banda: n.o de personas con los brazos arriba a < BAND m de la banda, dentro del largo del campo,
    maximo en +-1 s;
  - portero con el balon: alguien dentro de un area con las munecas juntas (< HANDS alturas de caja) entre la
    cadera y la cabeza; el equipo es el que defiende esa area.
Pseudoetiquetas: la entrega con pmax >= 0,9.

Puerta (la comun de T3-T4):
  - saque: AUC >= 0,75 para D frente a vivo, o lift de P(D | senal) >= 1,5, con cobertura >= 10 % de los
    tramos D predichos que empiezan junto a la banda;
  - portero: acuerdo del equipo con la prediccion segura >= 0,85, con cobertura >= 10 % de los frames dudosos
    o de los huecos del ATD dentro de un area.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from hackbarca.eval.evaluate import pm
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.pipeline.gate_shape import auc
from hackbarca.vision.video_pass import PERSON_EVERY

BAND = 3.0      # m a la banda
HANDS = 0.25    # distancia entre munecas / altura de la caja
HALF_WIN = 25   # +-1 s


def _data(*parts):
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), *parts)


def load_pose() -> np.ndarray:
    blocks = sorted(glob.glob(_data("derived", "pose_pass", "block*.npz")))
    rows = [np.load(f)["pose"] for f in blocks]
    return np.concatenate(rows) if rows else np.zeros((0, 7), dtype=np.float32)


def window_max(x: np.ndarray, grid: np.ndarray, half: int) -> np.ndarray:
    """Maximo de x en +-half frames, solo sobre los frames de la rejilla."""
    out = np.zeros_like(x)
    for d in range(-half, half + 1, PERSON_EVERY):
        out = np.maximum(out, np.roll(np.where(grid, x, 0), d))
    return out


def main() -> int:
    from hackbarca.io.load_match import load_match
    match = load_match()
    n, length, width = match.n_frames, match.pitch_length, match.pitch_width
    pose = load_pose()
    fr = pose[:, 0].astype(int)
    x, y, up, together, height = pose[:, 1], pose[:, 2], pose[:, 4], pose[:, 5], pose[:, 6]
    grid = np.zeros(n, dtype=bool)
    grid[np.unique(fr)] = True
    print("pose: %d personas en %d frames de la rejilla (%.1f por frame); con munecas %.3f; brazos arriba %.4f" % (
        len(pose), grid.sum(), len(pose) / max(grid.sum(), 1), np.isfinite(together).mean(), np.mean(up)))

    side = (y < BAND) | (y > width - BAND)
    throw_rows = (up > 0) & side & (x > 0) & (x < length)
    throw = window_max(np.bincount(fr[throw_rows], minlength=n).astype(float), grid, HALF_WIN)
    box_left = (x < 16.5) & (np.abs(y - width / 2) < 20.16)
    box_right = (x > length - 16.5) & (np.abs(y - width / 2) < 20.16)
    hold = (together < HANDS) & (height > 0.1) & (height < 0.9)
    hold_left = np.bincount(fr[hold & box_left], minlength=n) > 0
    hold_right = np.bincount(fr[hold & box_right], minlength=n) > 0

    pred = pm.label_codes(pm.read_label_csv(os.path.join("submissions", "La MasIA.csv")))
    pmax = np.load(_data("derived", "proba_l3.npz"))["proba"].max(1)
    graded = np.zeros(n, dtype=bool)
    home_left = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
        home_left[a:b] = np.nanmean(match.players_x[a:b][:, match.team_of_track == 0]) < length / 2
    on = grid & graded
    sure = on & (pmax >= 0.9) & np.isin(pred, (0, 1, 2))
    doubt = on & (pmax < 0.7)

    # saque de banda
    dead = pred[sure] == 2
    sig = throw[sure] > 0
    base = dead.mean()
    lift = dead[sig].mean() / base if sig.any() else np.nan
    print("saque: senal en %.3f de la rejilla segura | AUC D frente a vivo %.3f | P(D|senal) %.3f, base %.3f, lift %.2f" % (
        sig.mean(), auc(throw[sure], dead), dead[sig].mean() if sig.any() else np.nan, base, lift))
    # cobertura: tramos D predichos que empiezan junto a la banda (balon del ATD a < BAND m)
    starts, short_starts = [], []
    for r in GRADED:
        segs = pm.segments(pred, [tuple(r)])
        for i, (a, b, c) in enumerate(segs):
            by = match.ball_y[a]
            if c == 2 and 0 < i and np.isfinite(by) and (by < BAND + 2 or by > width - BAND - 2):
                starts.append(a)
                if b - a < 125 and i < len(segs) - 1:
                    short_starts.append(a)

    def covered(ts):
        ts = np.array(ts, dtype=int)
        return np.mean([throw[max(t - 75, 0):t + 75].max() > 0 for t in ts]) if len(ts) else np.nan
    print("saque: tramos D junto a la banda %d, con senal a <= 3 s %.3f | D cortos junto a la banda %d, con senal %.3f" % (
        len(starts), covered(starts), len(short_starts), covered(short_starts)))
    print("saque: senal en frames dudosos %.3f" % (throw[doubt] > 0).mean())

    # portero con el balon: el equipo que defiende el area
    # local (0) defiende la izquierda si home_left
    team = np.full(n, -1)
    one = hold_left ^ hold_right
    team[one & hold_left] = np.where(home_left, 0, 1)[one & hold_left]
    team[one & hold_right] = np.where(home_left, 1, 0)[one & hold_right]
    live_sure = sure & np.isin(pred, (0, 1))
    m = live_sure & (team >= 0)
    print("portero: senal en %.3f de la rejilla (ambas areas %.3f) | acuerdo con el equipo seguro %.3f (n=%d) | "
          "P(D|senal) %.3f frente a base %.3f" % (
              (team >= 0)[on].mean(), (hold_left & hold_right)[on].mean(), (team[m] == pred[m]).mean() if m.any() else np.nan,
              m.sum(), (pred[sure & (team >= 0)] == 2).mean(), base))
    gap = on & np.isnan(match.ball_x)
    print("portero: cobertura de dudosos %.3f, de huecos del ATD %.3f" % ((team[doubt] >= 0).mean(), (team[gap] >= 0).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
