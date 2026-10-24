"""Agregados de la forma del relleno de balon por video (solo cuantiles y medias)."""
import numpy as np
from hackbarca.io.load_match import load_match
from hackbarca.pipeline.atd_stats import GRADED
from hackbarca.vision.ball_detector import _sim_windows, _load_cands, _cache_path, auto_conf_ref
from hackbarca.vision.ball_track import link_candidates

def runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)

d = np.load("data/derived/video_ball.npz")
f, ok = d["frame"], ~np.isnan(d["x"])
cuts = np.flatnonzero(np.diff(f) != 1) + 1
on, off = [], []
for r in np.split(ok, cuts):
    on += list(runs(r)); off += list(runs(~r))
on, off = np.array(on), np.array(off)
print("huecos reales: tramos cubiertos %d, duracion p25/p50/p75/p90 %s frames, media %.0f" % (len(on), np.percentile(on, [25, 50, 75, 90]).round().astype(int).tolist(), on.mean()))
print("huecos reales: tramos sin cubrir %d, duracion p25/p50/p75/p90 %s frames, media %.0f" % (len(off), np.percentile(off, [25, 50, 75, 90]).round().astype(int).tolist(), off.mean()))

match = load_match()
graded = np.zeros(match.n_frames, bool)
for a, b in GRADED:
    graded[a:b] = True
cands = _load_cands(_cache_path())
ref = auto_conf_ref(cands)
errs = []
for a, b in _sim_windows(graded & ~np.isnan(match.ball_x), 40):
    fr = np.arange(a, b)
    path = link_candidates([cands.get(int(t), np.zeros((0, 3))) for t in fr], match.fps, ref)
    m = ~np.isnan(path[:, 0])
    errs += list(np.hypot(path[m, 0] - match.ball_x[fr][m], path[m, 1] - match.ball_y[fr][m]))
errs = np.array(errs)
print("huecos simulados, error en cubiertos: p25/p50/p75/p90/p95 %s m; >2 m %.3f; >5 m %.3f; >10 m %.3f" % (
    np.percentile(errs, [25, 50, 75, 90, 95]).round(2).tolist(), (errs > 2).mean(), (errs > 5).mean(), (errs > 10).mean()))
