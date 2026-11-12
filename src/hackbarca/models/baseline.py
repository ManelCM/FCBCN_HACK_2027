"""Baseline oficial (starter/baseline.py) sobre MatchTracking: vecino mas cercano + voto en ventana.

Mismo criterio que el oficial: sin balon no hay respuesta (NOBODY), y el suavizado deja D donde
nadie tiene el balon en toda la ventana. Se reutiliza `smooth` del repo oficial tal cual.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from hackbarca.io.load_match import MatchTracking

_STARTER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "external", "more-than-a-hack-2027", "starter")
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

from baseline import NOBODY, smooth  # noqa: E402

LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2}


def nearest_team(match: MatchTracking) -> np.ndarray:
    """Por frame, equipo (0/1) del track mas cercano al balon; NOBODY sin balon o sin jugadores."""
    dx = match.players_x - match.ball_x[:, None]
    dy = match.players_y - match.ball_y[:, None]
    dist = np.where(np.isnan(dx) | np.isnan(dy), np.inf, dx * dx + dy * dy)
    best = np.argmin(dist, axis=1)
    out = match.team_of_track[best].astype(np.int8)
    out[~np.isfinite(dist[np.arange(len(best)), best])] = NOBODY
    return out


def predict_baseline(match: MatchTracking, window: int = 50) -> np.ndarray:
    """Codigos H/A/D (0/1/2) por frame."""
    labels = smooth(nearest_team(match), window)
    return np.select([labels == "H", labels == "A"], [0, 1], 2).astype(np.int8)
