"""Analisis de errores de L3 en el banco (T8): donde se pierde el spellF1.

    uv run python -m hackbarca.pipeline.error_analysis

Usa las probabilidades fuera de muestra que guarda `tune_l3 --no-tags` y la decodificacion
congelada en el artefacto. Solo datos externos.
"""

from __future__ import annotations

import json
import os

import numpy as np

from hackbarca.eval.bench import pm, pooled
from hackbarca.io.degrade import degrade
from hackbarca.io.external import load_metrica_bench
from hackbarca.models.l3 import DecodeParams, decode
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES
from hackbarca.pipeline.train_l3_final import DECODE_PATH

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
H, A, D, X = 0, 1, 2, 3


def spell(gts, preds) -> float:
    return pooled(gts, preds)["spell_f1_mean"]


def shift_boundaries(pred: np.ndarray, k: int) -> np.ndarray:
    """Desplaza toda la secuencia k frames (k>0 = las fronteras llegan antes)."""
    return np.concatenate([pred[k:], np.repeat(pred[-1:], k)]) if k > 0 else \
        np.concatenate([np.repeat(pred[:1], -k), pred[:k or None]])


def long_gap_mask(ball_x: np.ndarray, min_len: int = 100) -> np.ndarray:
    missing = np.isnan(ball_x)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    out = np.zeros(len(ball_x), dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a > min_len:
            out[a:b] = True
    return out


def main() -> int:
    blob = np.load(os.path.join(_REPO_ROOT, "external_data", "oof_l3_notags.npz"))
    with open(DECODE_PATH) as handle:
        params = DecodeParams(**json.load(handle)["decode"])
    gts, probas, preds, gaps = [], [], [], []
    for i, game in enumerate(BENCH_GAMES):
        gt, proba = blob["gt%d" % i], blob["p%d" % i]
        match, _ = load_metrica_bench(game)
        gaps.append(long_gap_mask(degrade(match, BENCH_DEGRADE, seed=i).ball_x))
        gts.append(gt)
        probas.append(proba)
        preds.append(decode(proba, pm.valid_runs(gt), params))

    base = spell(gts, preds)
    print("L3 OOF (decodificacion congelada): spellF1={:.4f}".format(base))

    # --- oraculos ---
    perfect_d = [np.where(g == D, D, np.where(p == D, np.argmax(pr[:, :2], 1), p)).astype(np.int8)
                 for g, p, pr in zip(gts, preds, probas)]
    print("oraculo D perfecto (equipo del modelo):      {:.4f}".format(spell(gts, perfect_d)))
    perfect_team = [np.where((p != D) & (g != D) & (g != X), g, p).astype(np.int8)
                    for g, p in zip(gts, preds)]
    print("oraculo equipo perfecto (D del modelo):       {:.4f}".format(spell(gts, perfect_team)))
    # decodificar con el D verdadero impuesto: cuanto falla la segmentacion de equipo
    forced = []
    for g, pr in zip(gts, probas):
        q = pr.copy()
        q[g == D] = [1e-6, 1e-6, 1.0]
        q[(g == H) | (g == A), 2] = 1e-6
        forced.append(decode(q / q.sum(1, keepdims=True), pm.valid_runs(g), params))
    print("D verdadero impuesto + Viterbi de equipo:      {:.4f}".format(spell(gts, forced)))

    print("desplazamiento global de fronteras (frames, >0 = antes):")
    print("  " + "  ".join("{:+d}:{:.4f}".format(k, spell(gts, [shift_boundaries(p, k) for p in preds]))
                           for k in (-25, -12, -6, 0, 6, 12, 25)))

    # --- acierto por frame segun haya balon ---
    for name, sel in (("balon visible o hueco corto", lambda gp: ~gp), ("hueco largo (>4 s)", lambda gp: gp)):
        hits, total, dead_hits, dead_total = 0, 0, 0, 0
        for g, p, gp in zip(gts, preds, gaps):
            m = sel(gp) & (g != X)
            hits += np.sum(p[m] == g[m])
            total += m.sum()
            md = m & (g == D)
            dead_hits += np.sum(p[md] == D)
            dead_total += md.sum()
        print("{:<28} frames={:6d} acierto={:.3f} recall D={:.3f}".format(
            name, total, hits / max(total, 1), dead_hits / max(dead_total, 1)))

    # --- posesiones GT perdidas (sin pareja a IoU 0,5) por duracion ---
    buckets = ((0, 75), (75, 250), (250, 750), (750, 10 ** 9))
    found = {b: [0, 0] for b in buckets}
    for g, p in zip(gts, preds):
        runs = pm.valid_runs(g)
        gs, ps = pm.possession_spells(g, runs), pm.possession_spells(pm.fill_abstentions(p, runs), runs)
        overlaps = pm.spell_overlaps(gs, ps)
        flags = pm.matched_gt_flags(overlaps, len(gs), 0.5)
        for (start, stop, _), ok in zip(gs, flags):
            for b in buckets:
                if b[0] <= stop - start < b[1]:
                    found[b][0] += int(ok)
                    found[b][1] += 1
    print("posesiones GT emparejadas a IoU 0,5 por duracion:")
    for b, (ok, n) in found.items():
        print("  {:>5.0f}-{:<5} s: {:4d}/{:<4d} = {:.2f}".format(b[0] / 25, "inf" if b[1] > 1e8 else b[1] // 25,
                                                                ok, n, ok / max(n, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
