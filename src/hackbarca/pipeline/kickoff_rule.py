"""Regla del saque inicial de cada parte (ADR-0022): lo anterior al saque es balon muerto.

FAQ.md: la 2.a parte puntua desde el frame 110.808, pero el saque llega en el 113.742, y esos frames son
D. La entrega E090c sacaba ahi un 46 % de D y un 54 % de juego vivo (kickoff_report). La regla es general
y algoritmica, sin constantes del partido (ADR-0019):

1. el Smart Tagging marca cada saque inicial con SET PIECES / Type = "Kick Off (Start)", como una
   ventana de ~5 s que contiene el saque (lo comprueba starter/load_smart_tagging.check_time_base);
2. dentro de cada parte puntuada, todo lo anterior al inicio de esa ventana es D;
3. dentro de la ventana, el saque es el primer frame en que el balon del tracking se aleja mas de
   MOVE_M de su primera posicion vista en la ventana; si no se ve, se queda en el inicio de la ventana.
"""

from __future__ import annotations

import numpy as np

from hackbarca.io.load_match import MatchTracking

KICKOFF_CODE = "set pieces"
KICKOFF_TYPE = "kick off (start)"
MOVE_M = 1.0
D = 2


def load_events(data_dir=None):
    """Instancias del Smart Tagging real con el cargador oficial (lee data/, ADR-0004)."""
    import sys

    from hackbarca.io.smart_tagging import _STARTER
    if _STARTER not in sys.path:
        sys.path.insert(0, _STARTER)
    import load_smart_tagging as lst
    return lst.load(data_dir=data_dir)


def kickoff_frames(events, match: MatchTracking, graded) -> list:
    """Frame del saque inicial de cada parte puntuada (None si el Smart Tagging no lo marca)."""
    tags = sorted((e["start_frame"], e["end_frame"]) for e in events
                  if (e["code"] or "").strip().lower() == KICKOFF_CODE
                  and (e["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE)
    out = []
    for start, end in graded:
        inside = [(a, b) for a, b in tags if start - 250 <= a < end]
        if not inside:
            out.append(None)
            continue
        a, b = inside[0]
        x, y = match.ball_x[a:b], match.ball_y[a:b]
        seen = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        kick = a
        if len(seen):
            moved = np.hypot(x[seen] - x[seen[0]], y[seen] - y[seen[0]]) > MOVE_M
            if moved.any():
                kick = a + int(seen[np.argmax(moved)])
        out.append(kick)
    return out


def apply_kickoff_dead(codes: np.ndarray, kickoffs, graded) -> np.ndarray:
    out = codes.copy()
    for (start, end), kick in zip(graded, kickoffs):
        if kick is not None and kick > start:
            out[start:min(kick, end)] = D
    return out
