"""Ventanas de reanudacion (SET PIECES) del Smart Tagging, reales y simuladas para el banco.

Cada SET PIECES marca una ventana de ~5 s alrededor de la reanudacion, no el instante
(starter/load_smart_tagging.py). Son inferidas: faltan muchas (59 para >100 parones) y puede
haber falsos positivos.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_STARTER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "external", "more-than-a-hack-2027", "starter")

WINDOW = 125  # ~5 s a 25 fps


def set_piece_windows(game=None, data_dir=None) -> list[tuple[int, int]]:
    """[(start_frame, end_frame)] de los SET PIECES reales, en video frames. Lee data/ (ADR-0004)."""
    if _STARTER not in sys.path:
        sys.path.insert(0, _STARTER)
    import load_smart_tagging as lst

    events = lst.load(game, data_dir) if game else lst.load(data_dir=data_dir)
    return [(e["start_frame"], e["end_frame"]) for e in events
            if (e["code"] or "").strip().lower() == "set pieces"]


def simulate_set_piece_windows(events: pd.DataFrame, keep: float = 0.6, width: int = WINDOW,
                               seed: int = 0) -> list[tuple[int, int]]:
    """Ventanas tipo Smart Tagging a partir de los SET PIECE de un banco: se pierde un 1-keep y
    la reanudacion cae en una posicion aleatoria dentro de la ventana."""
    rng = np.random.default_rng(seed)
    out = []
    for restart in events.loc[events["type"] == "SET PIECE", "start"].to_numpy():
        if rng.random() >= keep:
            continue
        before = int(rng.integers(0, width + 1))
        out.append((int(restart) - before, int(restart) - before + width))
    return out
