import numpy as np

from hackbarca.vision.ball_track import link_candidates


def test_link_follows_true_ball_and_drops_false_positives():
    rng = np.random.default_rng(0)
    n = 200
    t = np.arange(n)
    true = np.column_stack([20 + 0.4 * t, 30 + 5 * np.sin(t / 20)])   # 10 m/s
    cands = []
    for f in range(n):
        rows = []
        if rng.random() < 0.8:   # el detector ve el balon el 80 % de las veces
            rows.append([*(true[f] + rng.normal(0, 0.2, 2)), rng.uniform(0.3, 0.9)])
        for _ in range(rng.integers(0, 3)):   # falsos positivos por todo el campo
            rows.append([rng.uniform(0, 105), rng.uniform(0, 68), rng.uniform(0.2, 0.6)])
        cands.append(np.array(rows).reshape(-1, 3))
    path = link_candidates(cands)
    found = ~np.isnan(path[:, 0])
    err = np.hypot(*(path[found] - true[found]).T)
    assert found.mean() > 0.7
    assert np.median(err) < 0.5
    assert np.mean(err > 5) < 0.05   # casi nunca se va a un falso positivo


def test_link_low_confidence_detector_needs_conf_ref():
    from hackbarca.vision.ball_detector import auto_conf_ref
    n = 100
    true = np.column_stack([20 + 0.4 * np.arange(n), np.full(n, 30.0)])
    cands = [np.array([[*true[f], 0.02]]) for f in range(n)]   # detector bien colocado pero poco seguro
    assert np.isnan(link_candidates(cands)[:, 0]).all()
    ref = auto_conf_ref(dict(enumerate(cands)))
    assert np.isclose(ref, 0.02)
    np.testing.assert_allclose(link_candidates(cands, conf_ref=ref), true)


def test_cands_cache_roundtrip(tmp_path):
    from hackbarca.vision.ball_detector import _load_cands, _save_cands
    cands = {3: np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 0.1]]), 7: np.zeros((0, 3)), 9: np.array([[5.0, 6.0, 0.9]])}
    path = str(tmp_path / "c.npz")
    _save_cands(cands, path)
    back = _load_cands(path)
    assert sorted(back) == [3, 7, 9]
    for f in cands:
        np.testing.assert_allclose(back[f], cands[f])


def test_link_empty_frames():
    assert np.isnan(link_candidates([np.zeros((0, 3))] * 5)).all()


def test_tiles_cover_full_hd():
    from hackbarca.vision.ball_detector import TILE, tiles
    corners = tiles(1920, 1080)
    covered = np.zeros((1080, 1920), bool)
    for x, y in corners:
        covered[y:y + TILE, x:x + TILE] = True
    assert covered.all()


def test_to_pitch_inverts_the_official_projection():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external",
                                    "more-than-a-hack-2027", "starter"))
    from load_homography import project
    from hackbarca.vision.ball_detector import to_pitch
    matrix = np.array([[1822.663, -511.2005, 44.48168], [1.698323, 169.2971, 255.7303],
                       [-0.01907188, -0.5269586, 1.0]])   # ejemplo publicado en DATA.md
    pitch = (105.0, 68.0)
    pts = np.array([[52.5, 34.0], [10.0, 5.0], [90.0, 60.0]])
    px, py = project(matrix, pts[:, 0] / pitch[0], pts[:, 1] / pitch[1])
    back = to_pitch(np.column_stack([px, py, np.ones(3)]), matrix, pitch)
    np.testing.assert_allclose(back[:, :2], pts, atol=1e-6)
