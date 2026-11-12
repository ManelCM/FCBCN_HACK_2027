"""Ajuste de L1 en el banco externo degradado: busqueda aleatoria + validacion dejando un partido fuera.

    uv run python -m hackbarca.pipeline.tune_l1 --trials 60

Nunca usa datos del reto. El numero que se reporta es el spellF1 agregado de los partidos
retenidos, cada uno puntuado con los parametros elegidos en los otros dos.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace

import numpy as np

from hackbarca.decode.viterbi import decode_runs, transition_costs
from hackbarca.eval.bench import pm, pooled
from hackbarca.io.degrade import degrade
from hackbarca.io.external import load_metrica_bench, metrica_events
from hackbarca.io.smart_tagging import simulate_set_piece_windows
from hackbarca.models.l1 import EPS, L1Params, apply_tags, interpolate_ball, team_distances
from hackbarca.pipeline.bench_config import BENCH_DEGRADE, BENCH_GAMES

GAPS = (10, 25, 50, 100)


def sample_params(rng) -> L1Params:
    return L1Params(
        max_gap=int(rng.choice(GAPS)),
        dist_scale=float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))),
        p_dead_visible=float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
        p_dead_gap=float(rng.uniform(0.3, 0.95)),
        switch=float(np.exp(rng.uniform(np.log(5), np.log(400)))),
        to_dead=float(np.exp(rng.uniform(np.log(5), np.log(200)))),
        from_dead=float(np.exp(rng.uniform(np.log(5), np.log(200)))),
        tag_before=int(rng.integers(25, 400)),
        p_tag_before=float(rng.uniform(0.3, 0.95)) if rng.random() < 0.75 else 0.0,
        p_tag_in=float(rng.uniform(0.05, 0.7)),
    )


class CachedGame:
    """Partido degradado con distancias precalculadas por max_gap (lo caro de las emisiones)."""

    def __init__(self, match, gt, windows=None):
        self.match, self.gt, self.runs = match, gt, pm.valid_runs(gt)
        self.windows = windows
        self.cache = {}

    def features(self, max_gap):
        if max_gap not in self.cache:
            bx, by, gap = interpolate_ball(self.match, max_gap)
            known = ~np.isnan(bx)
            d_home, d_away = team_distances(self.match, np.nan_to_num(bx), np.nan_to_num(by))
            m = self.match
            out = (bx < -1) | (bx > m.pitch_length + 1) | (by < -1) | (by > m.pitch_width + 1)
            diff = np.where(np.isfinite(d_home) & np.isfinite(d_away), d_away - d_home, 0.0)
            self.cache[max_gap] = (known, np.clip(diff, -50, 50), out)
        return self.cache[max_gap]

    def predict(self, p: L1Params):
        known, diff, out = self.features(p.max_gap)
        p_home = np.where(known, 1.0 / (1.0 + np.exp(-diff / p.dist_scale)), 0.5)
        p_dead = np.where(known, np.where(out, p.p_dead_out, p.p_dead_visible), p.p_dead_gap)
        p_dead = apply_tags(p_dead, self.windows, p)
        probs = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
        trans = transition_costs(p.switch, p.to_dead, p.from_dead)
        return decode_runs(-np.log(probs + EPS), trans, self.runs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    games = []
    for i, game in enumerate(BENCH_GAMES):
        match, gt = load_metrica_bench(game)
        windows = simulate_set_piece_windows(metrica_events(game), seed=i)
        games.append(CachedGame(degrade(match, BENCH_DEGRADE, seed=i), gt, windows))

    rng = np.random.default_rng(args.seed)
    trials = [L1Params()] + [sample_params(rng) for _ in range(args.trials)]
    # scores[t][g] = spellF1 del trial t en el partido g
    scores = np.zeros((len(trials), len(games)))
    preds = {}
    for t, params in enumerate(trials):
        for g, game in enumerate(games):
            preds[t, g] = game.predict(params)
            scores[t, g] = pm.evaluate(game.gt, preds[t, g])["spell_f1_mean"]
        print("trial {:3d} mean={:.4f} {}".format(t, scores[t].mean(), asdict(params)), flush=True)

    held_preds, chosen = [], []
    for g in range(len(games)):
        others = [o for o in range(len(games)) if o != g]
        best = int(np.argmax(scores[:, others].mean(axis=1)))
        chosen.append(best)
        held_preds.append(preds[best, g])
        print("retenido game {}: trial {} spellF1={:.4f}".format(BENCH_GAMES[g], best, scores[best, g]))
    result = pooled([gm.gt for gm in games], held_preds)
    print("L1 LOGO agregado: spellF1={:.4f} macroF1={:.4f} poss={} (gt {})".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]))
    best_all = int(np.argmax(scores.mean(axis=1)))
    print("mejor en los 3 (para usar en el reto): trial {} {}".format(best_all, asdict(trials[best_all])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
