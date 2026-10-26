"""SkillCorner open data (licencia MIT; github.com/SkillCorner/opendata) como datos extra de
entrenamiento (M3, E054). Liga australiana 2024/25, tracking de broadcast a 10 fps con jugadores
extrapolados fuera de plano y, por frame, **estado del balon (vivo/muerto) y equipo poseedor**: la
misma naturaleza que la GT de DFL y del reto (CHALLENGE.md).

Se pasa a 25 fps interpolando posiciones en el tiempo (solo entre frames a <= 0,3 s) y con la
etiqueta del frame original mas cercano, con un frame X entre partes como `dfl_to_bench`. Como en
Metrica y DFL limpios, el balon se borra en los frames muertos para que `degrade` (calibrado sobre el
tracking limpio) se aplique igual. Solo se usan para **entrenar**: el banco v3.1 (10 partidos) sigue
siendo el juez, asi que las notas siguen siendo comparables.

    uv run python -m hackbarca.io.skillcorner      # descarga y cachea los partidos con tracking
"""

from __future__ import annotations

import os

import numpy as np

from hackbarca.io.external import GT_A, GT_D, GT_H, GT_X, PITCH, _load_cache
from hackbarca.io.load_match import AWAY, HOME, MatchTracking

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE = os.path.join(_REPO_ROOT, "external_data", "skillcorner")
# partidos de matches.json del repositorio abierto; los que no traen tracking se saltan al cachear
MATCH_IDS = ("2017461", "2016236", "2015213", "2013725", "2011166", "2010085", "2007721", "2007448", "2006363",
             "2006229", "1996435", "1996436", "1986691", "1959846", "1953632", "1927964", "1925299", "1899585",
             "1886347", "1874553")
FPS = 25
MAX_INTERP_S = 0.3


def _resample(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Interpolacion lineal en el tiempo por columna; NaN si los vecinos faltan o distan > 0,3 s."""
    out = np.full((len(t_dst),) + values.shape[1:], np.nan)
    hi = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    lo = hi - 1
    w = np.clip((t_dst - t_src[lo]) / np.maximum(t_src[hi] - t_src[lo], 1e-9), 0, 1)
    close = (t_src[hi] - t_src[lo]) <= MAX_INTERP_S + 1e-9
    v_lo, v_hi = values[lo], values[hi]
    wv = w.reshape((-1,) + (1,) * (values.ndim - 1))
    res = v_lo * (1 - wv) + v_hi * wv
    ok = close.reshape((-1,) + (1,) * (values.ndim - 1)) & ~np.isnan(v_lo) & ~np.isnan(v_hi)
    out[ok] = res[ok]
    return out


def skillcorner_to_bench(dataset, pitch=PITCH):
    """(MatchTracking a 25 fps, gt) de un TrackingDataset de SkillCorner (coordinates="metrica")."""
    length, width = pitch
    home = dataset.metadata.teams[0]
    players = [p for team in dataset.metadata.teams for p in team.players]
    teams = np.array([HOME if p.team is home else AWAY for p in players], dtype=np.int8)
    parts, gts, periods = [], [], []
    for period in dataset.metadata.periods:
        recs = [r for r in dataset.records if r.period.id == period.id]
        if len(recs) < 2:
            continue
        t = np.array([r.timestamp.total_seconds() for r in recs])
        px = np.array([[r.players_data[p].coordinates.x if p in r.players_data and r.players_data[p].coordinates
                        else np.nan for p in players] for r in recs], dtype=float)
        py = np.array([[r.players_data[p].coordinates.y if p in r.players_data and r.players_data[p].coordinates
                        else np.nan for p in players] for r in recs], dtype=float)
        ball = np.array([[r.ball_coordinates.x, r.ball_coordinates.y] if r.ball_coordinates is not None
                         else [np.nan, np.nan] for r in recs], dtype=float)
        dead = np.array([str(r.ball_state).lower().endswith("dead") for r in recs])
        owner = np.array([GT_H if r.ball_owning_team is home else GT_A if r.ball_owning_team is not None else -1
                          for r in recs])
        label = np.where(dead, GT_D, owner)
        label[label < 0] = GT_X          # vivo sin poseedor: sin GT (se descarta al entrenar)
        grid = np.arange(t[0], t[-1], 1.0 / FPS)
        nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        prev = np.clip(nearest - 1, 0, len(t) - 1)
        nearest = np.where(np.abs(t[prev] - grid) < np.abs(t[nearest] - grid), prev, nearest)
        parts.append((_resample(t, px, grid) * length, _resample(t, py, grid) * width,
                      _resample(t, ball, grid) * np.array([length, width])))
        gts.append(label[nearest].astype(np.int8))
        periods.append(np.full(len(grid), period.id, dtype=np.int8))
    sep = lambda a, fill: np.full((1,) + a.shape[1:], fill, dtype=a.dtype)   # noqa: E731
    gt = np.concatenate(sum([[g, sep(g, GT_X)] for g in gts], [])[:-1])
    px = np.concatenate(sum([[p[0], sep(p[0], np.nan)] for p in parts], [])[:-1])
    py = np.concatenate(sum([[p[1], sep(p[1], np.nan)] for p in parts], [])[:-1])
    ball = np.concatenate(sum([[p[2], sep(p[2], np.nan)] for p in parts], [])[:-1])
    period = np.concatenate(sum([[p, sep(p, 0)] for p in periods], [])[:-1])
    ball[gt == GT_D] = np.nan
    n = len(gt)
    match = MatchTracking(n_frames=n, fps=FPS, pitch_length=length, pitch_width=width, team_of_track=teams,
                          players_x=px, players_y=py, players_speed=np.full_like(px, np.nan),
                          ball_x=ball[:, 0], ball_y=ball[:, 1], ball_speed=np.full(n, np.nan), period=period)
    return match, gt


def load_skillcorner_bench(match_id: str, cache_dir: str = CACHE):
    """(MatchTracking, gt) con cache .npz; descarga con kloppy la primera vez."""
    cache = os.path.join(cache_dir, "bench_sc{}.npz".format(match_id))
    if not os.path.exists(cache):
        from kloppy import skillcorner
        os.makedirs(cache_dir, exist_ok=True)
        dataset = skillcorner.load_open_data(match_id=match_id, only_alive=False, include_empty_frames=True,
                                             coordinates="metrica")
        match, gt = skillcorner_to_bench(dataset)
        np.savez_compressed(cache, gt=gt, **{k: v for k, v in vars(match).items()})
    return _load_cache(cache)


def available(cache_dir: str = CACHE) -> list[str]:
    return [m for m in MATCH_IDS if os.path.exists(os.path.join(cache_dir, "bench_sc{}.npz".format(m)))]


def main() -> int:
    for m in MATCH_IDS:
        try:
            match, gt = load_skillcorner_bench(m)
        except Exception as exc:     # partidos sin tracking en el repositorio abierto
            print(m, "sin tracking:", type(exc).__name__)
            continue
        lab = gt[gt != GT_X]
        print("{} frames {} | H {:.2f} A {:.2f} D {:.2f} X {:.2f} | balon visible {:.2f} | jugadores/frame {:.1f}".format(
            m, match.n_frames, np.mean(lab == GT_H), np.mean(lab == GT_A), np.mean(lab == GT_D), np.mean(gt == GT_X),
            np.mean(~np.isnan(match.ball_x)), float(np.median((~np.isnan(match.players_x)).sum(1)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
