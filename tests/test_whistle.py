import numpy as np

from hackbarca.audio.whistle import HOP, RATE, detect, half_offsets, to_tactical, tonal_score


def test_detects_synthetic_whistles_in_crowd_noise():
    rng = np.random.default_rng(0)
    seconds = 60
    audio = rng.normal(0, 0.1, RATE * seconds).astype(np.float32)   # ruido de grada
    t = np.arange(int(0.4 * RATE)) / RATE
    for start_s in (10.0, 31.0, 47.5):                                # tres pitidos de 0,4 s
        i = int(start_s * RATE)
        audio[i:i + len(t)] += 0.08 * np.sin(2 * np.pi * 3100 * t).astype(np.float32)
    events = detect(tonal_score(audio))
    starts = sorted(a for a, _ in events)
    assert len(starts) == 3
    for got, want in zip(starts, (10.0, 31.0, 47.5)):
        assert abs(got - want * RATE / HOP) <= 2   # frames de 40 ms


def test_half_offsets_and_mapping_to_tactical():
    n = 1000
    m = np.full(n, -1)
    m[100:400] = np.arange(100, 400) - 50      # 1.a parte: broadcast = tactico - 50
    m[600:900] = np.arange(600, 900) - 120     # 2.a parte: broadcast = tactico - 120
    m[150:160] = -1                            # repeticion sin pareja
    offsets = half_offsets(m, [(100, 400), (600, 900)])
    assert [o for _, _, o in offsets] == [50, 120]
    # (400, 405) cae en el descanso del broadcast (entre 350 y 480) y se descarta
    assert to_tactical([(60, 65), (400, 405), (700, 710)], offsets) == [(110, 115), (820, 830)]
