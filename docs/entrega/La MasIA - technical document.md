# La MasIA — Possession from automatic tracking (More Than A Hack 2027, Initial Challenge)

> DRAFT (2026-10-22). Final numbers are marked TODO and will be filled in when the model is frozen
> (14 Nov). To be exported as `La MasIA.pdf`.

## 1. Approach

We predict, for every tactical video frame, whether Barcelona (H) or Athletic (A) has possession or
the ball is dead (D). spellF1 rewards whole possessions with accurate boundaries, so the system has
three stages:

1. **A per-frame classifier** (LightGBM, 3 classes, 127 leaves) on ~65-110 hand-designed features
   computed from the automatic tracking. The features cover:
   - proximity of each team to the ball;
   - ball touches, with a per-match threshold;
   - who receives;
   - collective speed and team shape;
   - context around ball gaps;
   - TODO if adopted: change-of-possession signals and dead-ball layouts (section 4).
2. **A stacked second layer**: another LightGBM that sees the first layer's probabilities summarised
   1-10 s backwards, centred and forwards. It is trained on out-of-fold first-layer probabilities,
   so it learns how much to trust them.
3. **An offline decoder**: Viterbi over each half, with switch costs between H, A and D. It turns
   probabilities into possessions and places each boundary where the evidence on both sides is best
   explained, so there is no systematic lag. Argmax per frame scores ~0.2 spellF1 on our bench;
   decoded, it scores ~0.62.

**Why this rather than something else.** There is a single unlabeled match and no leaderboard, so
the decisive problem is *validation*, not model capacity. We built an external validation bench
(section 2) and fixed a selection protocol before running experiments. A change is kept only if it
improves consistently across held-out matches.

## 2. Validation without a leaderboard

- **Bench.** Ten open matches, all **degraded to look like the automatic tracking**: long ball gaps
  (lognormal, more likely at dead balls), filtered ball trajectories, tracks fragmented into 46
  anonymous slots, and position noise.
  - 3 Metrica Sports sample matches: labels derived from their events.
  - 7 DFL/IDSSE Bundesliga matches: labels taken from the **ball-status and ball-possession flags
    of their tracking**, the same nature as the official ground truth.
- **Calibration.** The simulator parameters were fixed so that:
  - the official baseline reproduces the eight published numbers of EVALUATION.md, including their
    ranking (smooth50 > smooth125 > smooth25 > raw);
  - aggregate statistics of the real tracking match: 40.7 % of graded frames without ball, gap
    length distribution and ball-dynamics percentiles.

  The same parameters, applied unchanged to DFL, give a baseline of 0.025 / 0.203 / 0.265 / 0.236
  (official: 0.049 / 0.200 / 0.243 / 0.224). Only aggregates were ever computed on the challenge
  data, always locally.
- **Controlled possession.** Event or tracking flags change team on every touch. CHALLENGE.md does
  not count a change unless the ball is *controlled*. We therefore score the bench with a rule: a
  steal counts only if the new team keeps the ball for at least T seconds (after a dead ball it
  always counts). T is chosen so that the bench's possession-duration distribution matches the
  public ground truth's: 17.6 % of possessions under 3 s, 14.2 s mean.
  - Metrica: T = 1.5 s gives 17.5 % and 13.7 s.
  - DFL: T = 2 s gives 17.5 %.

  T is never chosen by score.
- **Protocol.** Leave-one-match-out, pooled spellF1. A change is accepted only if it improves the
  pooled score by ≥ 0.01 *and* improves most held-out matches. The public 5-minute window
  (17 possessions, bootstrap noise ±0.38) is used only as a smoke test, never for selection.
- **Domain check.** Before accepting new features, we compare their percentiles on the real match and
  on the bench. This caught one feature set (fixed-threshold ball touches) that worked on the bench
  but not on the real, heavily filtered ball.

## 3. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | main input |
| Metrica open data, DFL/IDSSE open data | external validation and training bench |
| Smart Tagging | **dropped**: +0.008 on the bench (noise), and only 13 % of long ball gaps end with a tagged restart |
| broadcast ATD | **dropped as a ball source**: after aligning it to the tactical timeline (one constant offset per half), it fills only 7.5 % of the tactical ball gaps, because it loses the ball in the same plays |
| tactical video + homography | **dropped as a ball source**: an off-the-shelf detector (YOLOv8 COCO "sports ball", tiled, projected with the official homography, temporally linked) places the ball within 2 m on only 24 % of frames even where the tracking sees it |
| broadcast audio | **dropped**: a band-limited tonal whistle detector shows no signal around stoppages |

## 4. Possession rules implemented

- **Who owns the ball.** This is learned, not hand-coded. The inputs are:
  - proximity of the closest player of each team;
  - who touched the ball last and who touches it next, where touches are abrupt changes in ball
    velocity with a per-match threshold, so the definition is robust to how much the tracker smooths
    the ball;
  - who is closest to where the ball is going.
- **Loose balls, deflections and challenges.** The decoder's switch cost means possession only
  changes when the evidence for the other team is sustained, so a deflection or a duel without
  control does not flip it. In the training labels a CHALLENGE is never a change of possession.
  Completed long balls and switches of play are handled well (93-95 % frame accuracy); lost passes
  are the hard case.
- **Restarts and dead ball.** D is predicted from:
  - collective speed (players slow down at stoppages);
  - the ball near or beyond the lines;
  - long ball gaps;
  - TODO if adopted: dead-ball layouts (players still, crowded box, player on the touchline,
    goalkeeper in the six-yard box, kick-off formation).
- **Ball missing (41 % of graded frames).** Inside a gap the model relies on team context: who had
  the ball at the start and the end of the gap, team movement and collective speed.
- **No X on graded frames**; X only outside the two halves.

## 5. Limitations

- Bench labels come from events (Metrica) or from another provider's tracking flags (DFL), not from
  Metrica's corrected tracking. The controlled-possession rule approximates the official definition
  but is not identical.
- Undetected steals are the main loss: about half of the changes of team have no predicted boundary
  within 2 s.
- Inside long ball gaps, frame accuracy drops from ~0.92 to ~0.78.
- TODO: final numbers, and what we would do with more time (a ball detector fine-tuned on
  tracking-derived pseudo-labels, a learned audio event model, a sequence model).

## 6. Reproduction

```bash
git clone --recursive <repo> && cd <repo>
uv sync
# data bundle in data/ or MTAH_DATA_DIR pointing to it
uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"
```

The run is deterministic: the same data and the same commit give the same file, and the sha256 is
printed. The trained model is committed in `src/hackbarca/models/artifacts/`. It can be retrained
bit-for-bit from open data with `hackbarca.pipeline.train_l3_final`; LightGBM runs with
`deterministic=True` and forced column-wise histograms. TODO: final sha256 and commit.
