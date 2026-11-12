import numpy as np

from hackbarca.audio.whistle_v2 import RATE, _near, find_whistles, stationary_lines, whistle_score


def _synthetic(seconds=60, whistles=(12.0, 40.0), single_tone=25.0):
    rng = np.random.default_rng(0)
    t = np.arange(seconds * RATE) / RATE
    x = 0.05 * rng.standard_normal(len(t))
    x += 0.02 * np.sin(2 * np.pi * 3700 * t)  # linea fija de la mezcla
    for s in whistles:
        m = (t >= s) & (t < s + 0.6)
        x[m] += 0.03 * (np.sin(2 * np.pi * 3850 * t[m]) + np.sin(2 * np.pi * 4300 * t[m]))
    m = (t >= single_tone) & (t < single_tone + 0.6)
    x[m] += 0.06 * np.sin(2 * np.pi * 4100 * t[m])
    return x.astype(np.float32)


def test_two_tones_detected_single_tone_and_fixed_line_ignored():
    t, score, _ = whistle_score(_synthetic())
    found = find_whistles(t, score)
    assert len(found) == 2
    assert all(abs(a - s) < 0.2 for (a, _, _), s in zip(found, (12.0, 40.0)))


def test_stationary_line_reported():
    rep = stationary_lines(_synthetic(), step_s=2.0)
    assert any(abs(h - 3700) < 20 for h in rep["hz_fijos"])


def test_near_window():
    events = np.array([100, 500, 1000])
    refs = np.array([90, 1100])
    # evento - ref en [-1 s, +2 s] = [-25, +50] frames
    assert _near(events, refs, -1.0, 2.0).tolist() == [True, False, False]
