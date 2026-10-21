"""Alinea el tracking broadcast con el tactico y rellena el balon tactico donde falta.

El broadcast no esta sincronizado: tiene cortes, repeticiones y zooms (DATA.md). Pero ambos ATD
estan en coordenadas de campo, asi que cada plano del broadcast se puede situar en el timeline
tactico buscando el desfase que mejor hace coincidir a sus jugadores con los del mismo equipo.
Dentro de un plano el desfase es constante (los dos van a 25 fps).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hackbarca.io.load_match import MatchTracking

CAP = 5.0  # m: distancia maxima que cuenta en la puntuacion (robusto a jugadores sin pareja)


@dataclass
class Alignment:
    tactical_to_broadcast: np.ndarray        # (n_t,) indice de frame broadcast o -1
    segments: list                           # [(b_start, b_stop, offset, score)] aceptados
    rejected: int                            # planos descartados (mal encaje)
    flip: tuple                              # (flip_x, flip_y) aplicado al broadcast


def shot_segments(b: MatchTracking, min_players: int = 6, max_step: float = 1.0,
                  min_len: int = 25) -> list[tuple[int, int]]:
    """Planos [start, stop): tramos con jugadores suficientes y sin saltos de camara."""
    visible = ~np.isnan(b.players_x)
    enough = visible.sum(1) >= min_players
    step = np.hypot(np.diff(b.players_x, axis=0), np.diff(b.players_y, axis=0))
    both = visible[1:] & visible[:-1]
    smooth = (both & (step < max_step)).sum(1)
    # corte: menos de la mitad de los tracks del frame siguen y se mueven poco
    cut = smooth < 0.5 * np.maximum(visible[1:].sum(1), 1)
    ok = enough.copy()
    breaks = np.concatenate(([True], cut | ~enough[1:] | ~enough[:-1]))
    segments, start = [], None
    for f in range(b.n_frames):
        if breaks[f] or not ok[f]:
            if start is not None and f - start >= min_len:
                segments.append((start, f))
            start = f if ok[f] else None
    if start is not None and b.n_frames - start >= min_len:
        segments.append((start, b.n_frames))
    return segments


def _flipped(b: MatchTracking, flip) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px, py, bx, by = b.players_x, b.players_y, b.ball_x, b.ball_y
    if flip[0]:
        px, bx = b.pitch_length - px, b.pitch_length - bx
    if flip[1]:
        py, by = b.pitch_width - py, b.pitch_width - by
    return px, py, bx, by


def _score(t: MatchTracking, bx_frames, by_frames, b_team, offsets, sample) -> np.ndarray:
    """Puntuacion (m, menor es mejor) de cada desfase para los frames broadcast `sample`.

    Para cada jugador broadcast visible, distancia al jugador tactico mas cercano de su equipo
    en el frame b + desfase, recortada a CAP; mediana sobre jugadores y frames muestreados.
    """
    scores = np.full((len(offsets), len(sample)), np.nan)
    for j, f in enumerate(sample):
        vis = ~np.isnan(bx_frames[f])
        if vis.sum() < 3:
            continue
        frames = f + offsets
        valid = (frames >= 0) & (frames < t.n_frames)
        tx = t.players_x[np.clip(frames, 0, t.n_frames - 1)]   # (n_off, n_slots_t)
        ty = t.players_y[np.clip(frames, 0, t.n_frames - 1)]
        per_player = []
        for team in (0, 1):
            pb = vis & (b_team == team)
            if not pb.any():
                continue
            cols = t.team_of_track == team
            dx = tx[:, None, cols] - bx_frames[f, pb][None, :, None]
            dy = ty[:, None, cols] - by_frames[f, pb][None, :, None]
            d = np.sqrt(dx * dx + dy * dy)
            d = np.where(np.isnan(d), np.inf, d).min(axis=2)           # (n_off, n_pb)
            per_player.append(np.minimum(d, CAP))
        if per_player:
            s = np.median(np.concatenate(per_player, axis=1), axis=1)
            scores[valid, j] = s[valid]
    return np.nanmedian(np.where(np.isnan(scores), CAP, scores), axis=1)


def _best_offset(t, px, py, b_team, seg, candidates, n_sample=6):
    start, stop = seg
    sample = np.linspace(start, stop - 1, num=min(n_sample, stop - start)).astype(int)
    s = _score(t, px, py, b_team, candidates, sample)
    k = int(np.argmin(s))
    return int(candidates[k]), float(s[k])


def _drifts(t, px, py, b_team, seg, off, coarse, tol: int = 2) -> bool:
    """True si el primer y el ultimo tercio del plano piden desfases distintos (camara lenta,
    o un plano que mezcla dos momentos): entonces no hay un desfase unico valido."""
    start, stop = seg
    third = (stop - start) // 3
    if third < 10:
        return False
    span = max(coarse, third)
    cands = np.arange(off - span, off + span + 1)
    first, _ = _best_offset(t, px, py, b_team, (start, start + third), cands, n_sample=8)
    last, _ = _best_offset(t, px, py, b_team, (stop - third, stop), cands, n_sample=8)
    return abs(first - off) > tol or abs(last - off) > tol


def align(t: MatchTracking, b: MatchTracking, accept: float = 1.5, window: int = 3000,
          coarse: int = 10, flips=((False, False), (True, True))) -> Alignment:
    """Situa cada plano del broadcast en el timeline tactico."""
    segments = shot_segments(b)
    if not segments:
        return Alignment(np.full(t.n_frames, -1), [], 0, (False, False))
    all_offsets = np.arange(-b.n_frames, t.n_frames, coarse)

    # orientacion: la que mejor encaja el plano mas largo
    longest = max(segments, key=lambda s: s[1] - s[0])
    best_flip, best_flip_score = flips[0], np.inf
    for flip in flips:
        px, py, _, _ = _flipped(b, flip)
        _, s = _best_offset(t, px, py, b.team_of_track, longest, all_offsets)
        if s < best_flip_score:
            best_flip, best_flip_score = flip, s
    px, py, _, _ = _flipped(b, best_flip)

    mapping = np.full(t.n_frames, -1, dtype=np.int64)
    seg_score = np.full(t.n_frames, np.inf)
    accepted, rejected, previous = [], 0, None
    for seg in segments:
        cands = all_offsets
        if previous is not None:
            cands = np.arange(previous - window, previous + window + 1, coarse)
        off, s = _best_offset(t, px, py, b.team_of_track, seg, cands)
        if s > accept and previous is not None:
            off, s = _best_offset(t, px, py, b.team_of_track, seg, all_offsets)
        fine = np.arange(off - coarse, off + coarse + 1)
        off, s = _best_offset(t, px, py, b.team_of_track, seg, fine, n_sample=12)
        if s > accept or _drifts(t, px, py, b.team_of_track, seg, off, coarse):
            rejected += 1
            continue
        start, stop = seg
        tf = np.arange(start, stop) + off
        ok = (tf >= 0) & (tf < t.n_frames)
        tf, bf = tf[ok], np.arange(start, stop)[ok]
        # si otro plano ya reclamo el frame, se queda el de mejor encaje
        free = (mapping[tf] < 0) | (seg_score[tf] > s)
        mapping[tf[free]] = bf[free]
        seg_score[tf[free]] = s
        accepted.append((start, stop, off, s))
        previous = off
    return Alignment(mapping, accepted, rejected, best_flip)


def fuse_ball(t: MatchTracking, b: MatchTracking, alignment: Alignment) -> tuple[MatchTracking, np.ndarray]:
    """MatchTracking tactico con el balon rellenado desde el broadcast donde faltaba.

    Devuelve tambien `source` por frame: 0 = ATD tactico, 1 = broadcast, -1 = sin balon.
    """
    _, _, bbx, bby = _flipped(b, alignment.flip)
    ball_x, ball_y = t.ball_x.copy(), t.ball_y.copy()
    source = np.where(np.isnan(ball_x), -1, 0).astype(np.int8)
    m = alignment.tactical_to_broadcast
    fill = (source == -1) & (m >= 0)
    fill[fill] = ~np.isnan(bbx[m[fill]])
    ball_x[fill] = bbx[m[fill]]
    ball_y[fill] = bby[m[fill]]
    source[fill] = 1
    return replace(t, ball_x=ball_x, ball_y=ball_y), source


def alignment_report(t: MatchTracking, b: MatchTracking, alignment: Alignment, graded=None) -> dict:
    """Agregados para la puerta de decision (ADR-0004: nada por frame)."""
    graded = np.ones(t.n_frames, bool) if graded is None else graded
    _, _, bbx, bby = _flipped(b, alignment.flip)
    m = alignment.tactical_to_broadcast
    mapped = m >= 0
    t_missing = np.isnan(t.ball_x)
    b_ball = np.zeros(t.n_frames, bool)
    b_ball[mapped] = ~np.isnan(bbx[m[mapped]])
    both = mapped & ~t_missing & b_ball
    dist = np.hypot(t.ball_x[both] - bbx[m[both]], t.ball_y[both] - bby[m[both]])
    gap = t_missing & graded
    return {
        "planos_aceptados": len(alignment.segments),
        "planos_rechazados": alignment.rejected,
        "flip": alignment.flip,
        "frac_tactico_con_pareja": float(mapped[graded].mean()),
        "frac_huecos_rellenables": float((gap & b_ball).sum() / max(gap.sum(), 1)),
        "dist_balon_mediana_m": float(np.median(dist)) if len(dist) else float("nan"),
        "dist_balon_p90_m": float(np.percentile(dist, 90)) if len(dist) else float("nan"),
        "puntuacion_mediana_m": float(np.median([s for *_, s in alignment.segments])) if alignment.segments else float("nan"),
    }
