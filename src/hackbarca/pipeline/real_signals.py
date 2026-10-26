"""Senales que solo existen en el reto (broadcast, video, audio) y su puerta comun (T3-T4, E055+).

    scripts/run_aggregate.sh real_signals uv run python -m hackbarca.pipeline.real_signals

Cada senal es una serie por frame tactico (NaN = no se sabe) de dos tipos:
  - "dead": mas alto = mas probable que el balon este muerto (U1 plano no abierto del broadcast,
    U4 balon fuera o en un punto de reanudacion, A2 palabras de paron, A4 ritmo de habla, A6
    marcadores de tiempo);
  - "team": +1 local / -1 visitante (A1 jugador nombrado por el comentarista).
La puerta mide, con nuestra prediccion de la entrega y la GT publica de 5 min, solo agregados:
  - dead: AUC D frente a vivo donde el modelo esta seguro (pmax >= 0,9) y en la ventana publica, y
    el valor medio dentro de los D cortos (< 5 s) predichos frente a vivo seguro y D largo seguro:
    un arbitro independiente de si los D cortos (E041) son paron o juego;
  - team: acuerdo con el equipo predicho seguro, y cobertura y acuerdo donde el modelo duda o no hay
    balon (huecos), por desfase.
Puerta (docs/05_plan_mejora.md): AUC >= 0,75 o acuerdo >= 0,85, con cobertura >= 10 % de los frames
donde fallamos. Las series se guardan en data/derived/real_signals.npz (el agente no las lee).
"""

from __future__ import annotations

import json
import os

import numpy as np

from hackbarca.eval.evaluate import load_gt_codes, pm
from hackbarca.pipeline.atd_stats import GRADED, PUBLIC_GT

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H, A, D, X = 0, 1, 2, 3
FPS = 25
OUT = os.path.join(_REPO_ROOT, "docs", "investigacion", "resultados", "senales_reto.json")


def _data(*parts):
    return os.path.join(os.environ.get("MTAH_DATA_DIR", "data"), *parts)


def _offsets():
    from hackbarca.audio.whistle import half_offsets
    return half_offsets(np.load(_data("derived", "broadcast_alignment.npz"))["tactical_to_broadcast"], GRADED)


def _window_sum(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    n = len(x)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    a, b = np.clip(idx + lo, 0, n), np.clip(idx + hi + 1, 0, n)
    return cs[b] - cs[a]


# ----------------------------------------------------------------------------- senales

def broadcast_players(n_frames: int) -> np.ndarray:
    """U1: jugadores visibles en el broadcast en el mismo instante (plano abierto = juego). NaN fuera."""
    from hackbarca.io.load_match import load_match
    b = load_match(source="broadcast")
    count = (~np.isnan(b.players_x)).sum(1).astype(float)
    out = np.full(n_frames, np.nan)
    for lo, hi, offset in _offsets():          # tactico = broadcast + offset
        lo2, hi2 = max(lo, 0), min(hi, len(count))
        out[lo2 + offset:hi2 + offset] = count[lo2:hi2]
    return out


def ball_restart(match) -> dict:
    """U4: balon fuera del campo y cercania a puntos de reanudacion (con el balon del ATD)."""
    L, W = match.pitch_length, match.pitch_width
    x, y = match.ball_x, match.ball_y
    out_of = np.where(np.isnan(x), np.nan, ((x < -0.3) | (x > L + 0.3) | (y < -0.3) | (y > W + 0.3)).astype(float))
    spots = [(0, 0), (0, W), (L, 0), (L, W), (11, W / 2), (L - 11, W / 2), (L / 2, W / 2)]
    d_spot = np.min([np.hypot(x - sx, y - sy) for sx, sy in spots], axis=0)
    d_line = np.minimum.reduce([np.abs(x), np.abs(L - x), np.abs(y), np.abs(W - y)])
    step = np.full(len(x), np.nan)
    step[1:] = np.hypot(np.diff(x), np.diff(y))
    still = _window_sum(np.nan_to_num(step, nan=1.0) < 0.05, -12, 12) / 25
    return {"U4 balon fuera": out_of,
            "U4 balon quieto en punto de reanudacion": np.where(np.isnan(x), np.nan, (d_spot < 1.5) * still),
            "U4 balon quieto en una linea": np.where(np.isnan(x), np.nan, (d_line < 1.0) * still)}


def commentary(n_frames: int) -> dict:
    """A1, A2, A4, A6 a partir de la transcripcion de E033 (data/derived, el agente no la lee)."""
    from hackbarca.audio.commentary import _norm, to_frames, word_signals
    blob = np.load(_data("derived", "commentary_words.npz"))
    words = list(zip(blob["t"].tolist(), blob["w"].tolist()))
    offsets = _offsets()
    mentions, events = word_signals(words)
    out = {}
    # A1: una mencion en el frame f habla de lo que pasa 0-3 s antes -> la senal en t mira [t, t+lag]
    m_frames = to_frames([t for t, _ in mentions], offsets)
    team = np.zeros(n_frames)
    for f, s in zip(m_frames, [s for _, s in mentions]):
        if 0 <= f < n_frames:
            team[f] += s
    for lag in (50, 100):
        s = _window_sum(team, 0, lag)
        out["A1 equipo nombrado en los %d s siguientes" % (lag // FPS)] = np.where(s == 0, np.nan, np.sign(s))
    # A2: palabras de paron y de robo en los 5 s siguientes
    stop_kinds = ("foul", "free_kick", "corner", "throw_in", "offside", "goal_kick", "penalty", "card")
    for label, kinds in (("A2 palabras de paron", stop_kinds), ("A2 palabras de robo", ("turnover",))):
        ind = np.zeros(n_frames)
        for k in kinds:
            for f in to_frames(events[k], offsets):
                if 0 <= f < n_frames:
                    ind[f] = 1
        out[label + " en 5 s"] = (_window_sum(ind, 0, 125) > 0).astype(float)
    # A4: palabras por segundo en una ventana de 5 s (repeticiones y parones = mas charla, o menos)
    w_frames = to_frames([t for t, _ in words], offsets)
    rate = np.zeros(n_frames)
    for f in w_frames:
        if 0 <= f < n_frames:
            rate[f] += 1
    out["A4 palabras por segundo (5 s)"] = _window_sum(rate, -62, 62) / 5
    replay = np.zeros(n_frames)
    for (t, w), f in zip(words, w_frames):
        if 0 <= f < n_frames and _norm(w).strip() in ("replay", "again", "look"):
            replay[f] = 1
    out["A4 'replay/again/look' en 5 s"] = (_window_sum(replay, -62, 62) > 0).astype(float)
    # A6: marcadores de tiempo y de gol
    mark = np.zeros(n_frames)
    for (t, w), f in zip(words, w_frames):
        if 0 <= f < n_frames and _norm(w).strip() in ("half", "halftime", "whistle", "goal", "scores", "kickoff"):
            mark[f] = 1
    out["A6 marcador de tiempo o gol en 10 s"] = (_window_sum(mark, -125, 125) > 0).astype(float)
    return out


# ----------------------------------------------------------------------------- puerta

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    n1, n0 = float(len(pos)), float(len(neg))
    return float((ranks[:len(pos)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def contexts(pred, pmax, known, public):
    graded = np.zeros(len(pred), dtype=bool)
    for a, b in GRADED:
        graded[a:b] = True
    live = graded & ((pred == H) | (pred == A))
    short = np.zeros(len(pred), dtype=bool)
    long_d = np.zeros(len(pred), dtype=bool)
    for a, b in GRADED:
        segs = pm.segments(pred, [(a, b)])
        for n, (s, e, c) in enumerate(segs):
            if c == D:
                (short if e - s < 125 and 0 < n < len(segs) - 1 else long_d)[s:e] = True
    sure = pmax >= 0.9
    return dict(graded=graded, live_sure=live & sure, dead_sure=long_d & sure, short_d=short,
                unsure=graded & (pmax < 0.7), gap=graded & ~known,
                pub_live=(public == H) | (public == A), pub_dead=public == D)


def gate_dead(name, s, ctx) -> dict:
    row = dict(kind="dead",
               auc_seguro=auc(s[ctx["dead_sure"]], s[ctx["live_sure"]]),
               auc_publica=auc(s[ctx["pub_dead"]], s[ctx["pub_live"]]),
               media_vivo_seguro=float(np.nanmean(s[ctx["live_sure"]])),
               media_D_largo_seguro=float(np.nanmean(s[ctx["dead_sure"]])),
               media_D_cortos=float(np.nanmean(s[ctx["short_d"]])),
               cobertura_D_cortos=float(np.mean(np.isfinite(s[ctx["short_d"]]))),
               cobertura_dudosos=float(np.mean(np.isfinite(s[ctx["unsure"]]))))
    lo, hi = row["media_vivo_seguro"], row["media_D_largo_seguro"]
    # 0 = los D cortos se parecen al juego vivo seguro, 1 = a los parones seguros
    row["D_cortos_entre_vivo0_y_paron1"] = float((row["media_D_cortos"] - lo) / (hi - lo)) if hi != lo else float("nan")
    return row


def gate_team(name, s, ctx, pred, public) -> dict:
    team = np.where(pred == H, 1.0, np.where(pred == A, -1.0, np.nan))
    pub = np.where(public == H, 1.0, np.where(public == A, -1.0, np.nan))

    def agree(mask, ref):
        m = mask & np.isfinite(s) & np.isfinite(ref)
        return float(np.mean(s[m] == ref[m])) if m.any() else float("nan"), float(m.sum())

    a_sure, n_sure = agree(ctx["live_sure"], team)
    a_uns, n_uns = agree(ctx["unsure"], team)
    a_gap, n_gap = agree(ctx["gap"], team)
    a_pub, n_pub = agree(ctx["pub_live"], pub)
    return dict(kind="team", acuerdo_seguro=a_sure, n_seguro=n_sure, acuerdo_dudoso=a_uns,
                cobertura_dudosos=float(np.mean(np.isfinite(s[ctx["unsure"]]))), acuerdo_hueco=a_gap,
                cobertura_huecos=float(np.mean(np.isfinite(s[ctx["gap"]]))), acuerdo_publica=a_pub, n_publica=n_pub,
                cuota_local_predicha=float(np.nanmean(team[ctx["live_sure"]] == 1)))


def main() -> int:
    from hackbarca.io.load_match import load_match
    match = load_match()
    n = match.n_frames
    pred = pm.label_codes(pm.read_label_csv(os.path.join(_REPO_ROOT, "submissions", "La MasIA.csv")))
    pmax = np.load(_data("derived", "proba_l3.npz"))["proba"].max(1)
    public = load_gt_codes(PUBLIC_GT)
    ctx = contexts(pred, pmax, ~np.isnan(match.ball_x), public)
    signals = {"U1 jugadores visibles en el broadcast (menos = paron)": -broadcast_players(n)}
    signals.update(ball_restart(match))
    signals.update(commentary(n))
    np.savez_compressed(_data("derived", "real_signals.npz"), **{k: v.astype(np.float32) for k, v in signals.items()})
    report = {}
    for name, s in signals.items():
        report[name] = (gate_team(name, s, ctx, pred, public) if name.startswith("A1") else gate_dead(name, s, ctx))
        print("\n" + name)
        for k, v in report[name].items():
            print("   {:<34} {}".format(k, round(v, 3) if isinstance(v, float) else v))
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
