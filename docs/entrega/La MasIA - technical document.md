# La MasIA — Possession from automatic tracking (More Than A Hack 2027, Initial Challenge)

> DRAFT (2026-10-21). Final numbers are marked TODO and will be filled in when the model is frozen
> (14 Nov). To be exported as `La MasIA.pdf`.

## 1. Approach

We predict, for every tactical video frame, whether Barcelona (H) or Athletic (A) has possession or
the ball is dead (D). spellF1 rewards whole possessions with accurate boundaries, so we split the
problem into two parts:

1. **A per-frame classifier** (LightGBM, 3 classes) on ~65 hand-designed features computed from the
   automatic tracking: proximity of each team to the ball, ball touches, collective speed, team
   shape, and context around ball gaps.
2. **An offline decoder** (Viterbi with switch costs between H, A and D) that turns per-frame
   probabilities into possessions. It removes flicker and places each boundary where the evidence
   on both sides is best explained, so there is no systematic lag. Decoding is essential: argmax
   per frame scores ~0.1 spellF1 on our validation bench, against ~0.5 once decoded.

**Why this rather than something else.** There is a single unlabeled match and no leaderboard, so
the decisive problem is *validation*, not model capacity. We built an external validation bench
(section 2), fixed a selection protocol before running experiments, and kept models that improve
consistently across held-out matches. TODO: final model description.

## 2. Validation without a leaderboard

- **Bench.** Metrica Sports open sample data (3 matches, ~1,140 possessions), with per-frame H/A/D
  labels derived from their event data, **degraded to look like the automatic tracking**: long ball
  gaps (lognormal, mostly at dead balls), filtered ball trajectories, tracks fragmented into 46
  anonymous slots, and position noise.
- **Calibration.** The simulator parameters were chosen so that (a) the official baseline
  reproduces the eight published numbers of EVALUATION.md, including their ranking
  (smooth50 > smooth125 > smooth25 > raw), and (b) aggregate statistics of the real tracking match:
  40.7 % of graded frames without ball, gap length distribution, and ball-dynamics percentiles.
  Only aggregates were ever computed on the challenge data, and always locally.
- **Protocol.** Leave-one-match-out, pooled spellF1. A change is accepted only if it improves the
  pooled score by ≥ 0.01 *and* improves every held-out match. The public 5-minute window
  (17 possessions) is used only as a smoke test, never for selection.

## 3. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | main input |
| tactical homography | ball detection from video (TODO if adopted) |
| broadcast ATD | ball gap filling after timeline alignment (TODO if adopted) |
| tactical video | local ball detector, open pretrained weights, no hand labels (TODO if adopted) |
| Smart Tagging | **dropped**: +0.008 on the bench (noise), and only 13 % of long ball gaps end with a tagged restart |
| Metrica open data, DFL/IDSSE open data | external validation and training bench |

## 4. Possession rules implemented

- **Who owns the ball.** Learned, not hand-coded: proximity of the closest player of each team,
  plus who touched the ball last and who touches it next (touches = abrupt changes in ball
  velocity, with a per-match threshold so that the definition is robust to how much the tracker
  smooths the ball).
- **Loose balls and challenges.** The decoder's switch cost means possession only changes when the
  evidence for the other team is sustained: a deflection or a duel without control does not flip
  it. In the training labels a CHALLENGE is never a change of possession; a RECOVERY, a PASS or a
  CARRY by the other team is.
- **Restarts and dead ball.** D is predicted from collective speed (players slow down at
  stoppages: 1.1 vs 2.0 m/s in the public window), the ball near or beyond the lines, and long
  ball gaps.
- **Ball missing (41 % of graded frames).** Inside a gap the model relies on team context: who had
  it at the start and the end of the gap, team movement and collective speed. TODO: ball fusion.
- **No X on graded frames**; X only outside the two halves.

## 5. Limitations

- The bench labels come from events, not from corrected tracking; frame-level boundary
  definitions may differ slightly from the official ground truth.
- Very short possessions (< 3 s) are mostly missed (6 % recovered on the bench).
- Inside long ball gaps, frame accuracy drops from ~0.89 to ~0.74.
- TODO: final numbers and what we would do with more time (more matches, sequence model, video).

## 6. Reproduction

```bash
git clone --recursive <repo> && cd <repo>
uv sync
# data bundle in data/ or MTAH_DATA_DIR pointing to it
uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"
```

Deterministic: same data and same commit give the same file (sha256 printed). The trained model is
committed in `src/hackbarca/models/artifacts/` and can be retrained from open data with
`hackbarca.pipeline.train_l3_final`. TODO: final sha256 and commit.
