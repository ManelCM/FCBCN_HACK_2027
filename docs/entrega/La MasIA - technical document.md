# La MasIA — Possession from automatic tracking (More Than A Hack 2027, Initial Challenge)

> Final version (2026-11-12). The submitted file `La MasIA.csv` is regenerated bit for bit by the
> command in section 9 (three runs, same sha256 prefix `2223250da5edc7ed`).

## 1. Approach in one paragraph

For every tactical video frame we predict whether Barcelona (H) or Athletic (A) has possession or the
ball is dead (D). spellF1 scores whole possessions with tight boundaries, so the system is built around
sequences, not frames:

1. **Per-frame features** from the automatic tracking only: who is near the ball and for how long,
   ball touches, team shape and collective speed, restart layouts, the collective signature of a steal,
   and context around the gaps where the ball is missing.
2. **A stacked LightGBM**. A first layer predicts H/A/D per frame. A second layer re-reads the first
   layer's out-of-fold probabilities summarised 1-10 s backwards, centred and forwards, so it learns
   how far to trust them. Training gives **every possession the same total weight**, so short
   possessions (the ones spellF1 punishes when missed) count as much as long ones.
3. **A player-possessor network blended in logit space** (30 %): attention between the visible players
   of each frame plus a bidirectional GRU over 10 s, which estimates *which player* has the ball and
   derives the team from it (section 3.3).
4. **An offline Viterbi decoder** per half, with switch costs between H, A and D. It turns
   probabilities into possessions and places each boundary where the evidence on both sides is best
   explained, so there is no systematic lag.
5. **Two post-processing rules**: a predicted dead ball shorter than 3 s between two live spells is
   given to the previous spell (section 3.4), and everything before each half's kick-off is D
   (section 4).

The model is trained and tuned **only on open external data**. Nothing is trained, tuned or calibrated
on the challenge's ground truth or on the challenge's video (section 7).

**Why this rather than something else.** There is one unlabeled match and no leaderboard, so the
decisive problem is *validation*, not model capacity. We built an external bench (section 2), fixed a
selection protocol before running experiments, and accepted a change only when it improved
consistently across held-out matches. More than 110 experiments are logged; most of them were rejected,
and section 5 lists what was tried and dropped.

## 2. Validation without a leaderboard

### 2.1 Bench
- **Ten open matches**: 3 Metrica Sports sample matches (labels derived from their events) and 7
  DFL/IDSSE Bundesliga matches (labels from the **ball-status and ball-possession flags of their
  tracking**, the same nature as the official ground truth).
- **Degraded to look like the challenge's automatic tracking**: long ball gaps (lognormal, more likely
  around dead balls), a filtered ball trajectory, tracks fragmented into 46 anonymous slots, and
  position noise.
- **Calibration targets** (aggregates of the challenge's *tracking*, computed locally; never its ground
  truth):
  - the official baseline reproduces the eight published numbers of EVALUATION.md and their ranking;
  - the real tracking's share of graded frames without ball (40.7 %), the gap-length distribution,
    ball-dynamics percentiles (residual, velocity changes, touches per second) and the players'
    collective speed inside ball gaps compared with frames with ball;
  - **bench v3.1** (our final bench) also matches the rate of *slow live ball*. The first simulator
    added correlated noise that made a still ball look like it was moving at ~4 m/s, so on the bench a
    slow ball always meant a dead ball. The model learned that shortcut and produced about six times
    more short dead-ball spells on the challenge than on the bench. Fixing it moved slow live-ball runs
    from 0.006 to 0.96 per minute (clean tracking: 1.08) and made the bench show the defect.

### 2.2 Controlled possession
Event and tracking flags change team on every touch, while CHALLENGE.md only counts a change when the
new team *controls* the ball. The bench is therefore scored with a rule: a steal counts only if the
new team keeps the ball for at least T seconds (after a dead ball it always counts). T = 1.5 s for
Metrica and 2 s for DFL. T was chosen to understand what "controlled" means, never by score, and it is
a property of the bench labels, not of the model. Section 6 explains how the public window was used
for it, before the 3 Nov clarification.

### 2.3 Selection protocol (fixed before the experiments)
- Leave-one-match-out over the ten matches, pooled spellF1.
- A change is accepted only if it improves the pooled score by **≥ 0.01** *and* improves **at least 7
  of the 10** held-out matches.
- **Shape check on the challenge** (automatic pass/fail veto, no parameter is fitted with it). Aggregates
  of our own prediction on the real match, with no ground truth. A candidate that passes the bench is
  submitted only if its prediction meets fixed limits on the known defect of the challenge: at most 0.25
  dead-ball spells under 5 s per minute, 0.8-1.3 dead-ball spells per minute, and 38-50 % of D. Among the
  candidates that pass, the best on the bench wins, with the order written down before the challenge is
  looked at. The check never chooses between candidates. (Until 12 Nov the veto was relative to the
  previous submission; every candidate that improved the bench lowered the direct H↔A changes, so we
  moved to absolute limits on the dead-ball measures and report the direct changes as a known risk.)
- **Domain check**: before accepting a feature family we compare its percentiles on the real tracking
  and on the bench. This caught a fixed-threshold touch detector that worked on the bench and not on
  the heavily filtered real ball.

### 2.4 Progression on bench v3.1 (leave-one-match-out, pooled spellF1)
| model | spellF1 | matches improved |
|---|---|---|
| model trained on the previous bench (v3), decoded as submitted | 0.7010 | — |
| retrained on bench v3.1 | 0.7254 | 10 / 10 vs. the line above |
| + steal signatures (section 3.1) | **0.7410** | 9 / 10 vs. the line above |
| + equal weight per possession | **0.7555** | 8 / 10 vs. the line above |

The submitted pipeline adds the short dead-ball rule and the possessor blend; it is scored the same way,
with the rule inside: steal signatures 0.7389 → + equal weight per possession 0.7513 → **+ possessor blend
0.7621** (8 / 10 vs. the line before).

For reference, the official baseline scores 0.2433 on the public window and our first rule-based
model scored ~0.36 on the original bench. The bench is still somewhat easier than the challenge, so
we expect a lower score on the real match.

## 3. The model

### 3.1 Features (over 200 per frame)
Seven families, all computed per frame from the tracking, in home/away pairs so that the model can be
trained on every match twice with the teams swapped:
- **touch**: abrupt changes in ball velocity with a per-match threshold, who touched last and who
  touches next, time since and until a touch;
- **ball**: progression before and after, long flights, still ball;
- **players**: distance of the closest player of each team, who is closest to where the ball is
  going, collective speed;
- **blocks**: length, width, area and lines of each team, relative to the match;
- **structure**: players behind the ball, in the opponent's half, shape opening or closing;
- **restart**: dead-ball layouts (players still, crowded box, player on the touchline, goalkeeper in
  the six-yard box, kick-off formation);
- **steal signatures**: what a coach sees when a team wins the ball: its centroid moves forward and the
  other one drops back, the widest players open up and sprint forward. Measured per team along its own
  attacking axis. It attacks the steals that the ball alone does not show.

Every family is described with its change *after minus before* over 1-3 s windows: after a steal, the
change is informative while the current level is not.

### 3.2 Layers and training
- First layer: LightGBM, 3 classes, 127 leaves, 400 rounds, deterministic mode.
- Second layer: LightGBM on the first layer's out-of-fold probabilities with 1-10 s temporal context,
  plus the original features.
- Trained on the ten bench matches with their raw labels. The controlled-possession rule is used to
  score, not to train: training on it did not help.
- **Equal weight per possession**: each live frame weighs 8 s divided by the length of its possession in
  the controlled-possession labels, clipped to [0.25, 4], and the live weights are rescaled so that their
  sum is unchanged. Dead-ball frames weigh 1. Same weights on the team-swapped copy and in both layers.
  Alone it moves the bench from 0.7410 to 0.7555 (8 / 10).

### 3.3 Player-possessor network
- Input per frame (5 fps): the visible player slots (position, velocity, team, distance to the ball) and
  global context. Attention between players, then a bidirectional GRU over 10 s windows.
- Two heads: H/A/D, and "which player controls the ball" (or nobody), whose target comes from the clean
  bench tracking (the nearest player of the possessing team within 2 m of the clean ball). P(H) and P(A)
  combine the two heads; the prediction is averaged with the team-swapped input.
- Trained on the ten bench matches (6 epochs, fixed seed). Its weights are versioned (`possessor.pt`,
  0.8 MB). It runs on CPU with deterministic algorithms at submission time.
- Averaged with the LightGBM it did not help (0.7236 vs. 0.7254). **Blended in logit space** with weight
  0.3 (p ∝ p_LGBM^0.7 · p_possessor^0.3), chosen leaving one match out, it adds +0.011 with the short
  dead-ball rule (8 / 10).

### 3.4 Decoder
Viterbi per half. The costs are chosen on the bench with the same leave-one-match-out search as the
model (40 sampled settings), for the final blend and with the short dead-ball rule inside: temperature
1.02, switch 35.8, to-dead 165.6, from-dead 11.0. Opening a dead ball is expensive and closing it is
cheap: a new dead-ball spell needs strong evidence, which removes the short dead balls that split
possessions on the challenge. A duration model (HSMM), a minimum duration of 2 s, snapping boundaries to
the nearest touch, or making a direct change cheaper than going through a dead ball did not help.

### 3.5 Short dead-ball rule
On the challenge our prediction had about 0.68 dead-ball spells under 5 s per minute; the bench ground
truth has 0.09. Most of them split one possession in two (H D H), which costs two possessions in
spellF1. The rule (a D under 3 s between two live spells takes the previous label) was fixed on the
bench, where it is neutral on clean predictions (+0.001) and recovers +0.023 when the challenge's
excess is injected. On the challenge, the shape of our prediction moved as follows (aggregates of our
own prediction, no ground truth; the first two columns are earlier tracking-only versions, **without**
the short dead-ball rule or the kick-off rule; the third adds the steal signatures and both rules; the
last one is the submitted model):

| per minute of graded play | bench v3, no rules | bench v3.1, no rules | + steal signatures + both rules | **submitted** | limit |
|---|---|---|---|---|---|
| dead-ball spells under 5 s | 0.680 | 0.529 | 0.274 | **0.142** | ≤ 0.25 |
| dead-ball spells | 1.775 | 1.605 | 1.360 | **1.039** | 0.8-1.3 |
| direct H↔A changes | 1.397 | 1.482 | 1.539 | **1.435** | reported |
| share of D | 0.448 | 0.428 | 0.436 | **0.442** | 0.38-0.50 |

The submitted model predicts 2.46 possessions per minute (mean 13.6 s); the bench ground truth has 3.03.
It changes team directly less often than the bench ground truth, mostly inside ball gaps (section 8).

## 4. Possession rules implemented

- **Who owns the ball** is learned, not hand-coded, from proximity, touches, who receives, team shape
  and the collective signature of a steal.
- **Loose balls, deflections, duels.** Training labels never change team on a challenge, and the
  bench is scored with controlled possession. The decoder's switch cost then requires sustained
  evidence for the other team, so a deflection or a duel without control does not flip possession.
- **Restarts and dead ball.** D comes from collective speed, restart layouts, the ball near or beyond
  the lines, and the context of long ball gaps. Short dead balls inside play follow sections 3.4 and 3.5.
- **Ball missing (41 % of graded frames).** Inside a gap the model relies on team context: who had
  the ball before and after the gap, team movement and collective speed.
- **Before each kick-off.** The second half is graded from about two minutes before its kick-off, and
  those frames are D. The model alone predicted only 46 % D there. A general rule, with no
  match-specific constant, sets everything before each half's kick-off to D. The kick-off comes from
  the Smart Tagging instance `Kick Off (Start)`, a ~5 s window that contains it, used as an input.
  Inside that window, the kick-off is the first frame where the tracked ball moves more than 1 m.
  Compared with the kick-off frames published in the FAQ, this detection is 0.16 s late in the first
  half and 3.2 s early in the second. With the rule, 97.3 % of the frames before the second-half
  kick-off are D.
- **No X on graded frames.** X appears only outside the two halves.

## 5. Assets used and dropped

| asset or idea | status | evidence (bench v3.1 unless stated) |
|---|---|---|
| tactical tracking (players + ball) | **used** | main input |
| Metrica and DFL/IDSSE open data | **used** | validation and training bench |
| Smart Tagging | **used only for the kick-off rule** (section 4) | as model features it gave +0.008 (noise), and only 13 % of long gaps end with a tagged restart |
| coach-style family: steal signatures | **used** | +0.016, 9/10 |
| tactical video: who carries the ball (ball detector fine-tuned on crops of the challenge video) | **withdrawn for compliance** (section 7) | it was our best signal (0.7692, +0.044, 10/10), but its detector was trained on the challenge video |
| broadcast tracking as a ball source | dropped | fills 7.5 % of the tactical gaps: it loses the ball in the same plays |
| generic ball detector (YOLO COCO) | dropped | ball within 2 m on 24 % of frames |
| broadcast shot type (wide vs. cut) | dropped | 0.7311 vs. 0.7254, 6/10 |
| broadcast ball-action spotting (SoccerNet T-DEED, pretrained, not retrained) | dropped | +0.005, 6/10 |
| broadcast audio: whistle, commentary, crowd, kick sound | dropped | whistle heard at only ~10 % of stoppages; other cues AUC 0.47-0.55 |
| pose (throw-ins, goalkeeper with the ball), referee position | dropped | did not pass their gates |
| other coach-style families: formation, game situations, pressing, territory, game state | dropped | below the gate |
| sustained control (40 columns: closest-player persistence, support, collective deceleration) | dropped | 0.7296 vs. 0.7254, 4/10 |
| equal weight per possession | **used** | +0.0145, 8/10 |
| player-possessor network (PathCRF-like), blended in logit space | **used** | +0.011 with the short dead-ball rule, 8/10 (averaged in probability: 0.7236 vs. 0.7254) |
| MS-TCN, averaged or blended in logit space | dropped | averaged 0.7207 vs. 0.7254; in logit 0.7514 vs. 0.7555 |
| XGBoost blend, bagging | dropped | blend −0.008; bagging −0.011 |
| boundaries snapped to the nearest touch | dropped | −0.0004 to −0.013: the bench touch detector is too noisy |
| 20 SkillCorner open matches as extra training data (all labels) | dropped | 0.6856 vs. 0.7254 (worse in 9/10). SkillCorner has no dead-ball label |
| decoder: direct change cheaper than stop + restart | dropped | ≤ +0.005 |
| minimum possession of 2 s (post-processing) | dropped | +0.005, 6/10; it also removes real short possessions |
| two heads: P(live) and P(home \| live) | dropped | 0.7549 vs. 0.7555, 5/10; with the possessor blend 0.7603 vs. 0.7621 |
| blending more models in logit space (two-head LightGBM, MS-TCN, other possessor weights) | dropped | −0.002 to −0.005 vs. the submitted blend, 2-4/10 |
| simulated dead ball carried to the restart spot (extra training copies) | dropped | 0.7368 vs. 0.7555, worse |
| possessor with other seeds, averaged seeds, 10 epochs, separate weights for live/team or ball visible/in a gap | dropped | 0.7536-0.7615 vs. 0.7621 |
| SkillCorner for "who" only (with two heads) | not finished | stopped at the deadline; not used |

Equal weight per possession, the logit blend with a sequence model, boundaries snapped to touches, a
cheaper direct change in the decoder and a minimum possession came from reading other participants'
public repositories after the 3 Nov clarification. We re-implemented them on our bench and judged them
with the same protocol.

## 6. Public window and labelled data

**Public 5-minute window.** Following the rules clarified on 22-23 Oct and 3 Nov, it is used only as a
smoke test (format, nothing broken). It never selects, tunes or gates anything. Earlier uses, all
documented in our decision log (ADR-0019, ADR-0023):
- the possession-duration shape behind T (section 2.2);
- a second criterion for a broadcast signal that was later dropped;
- one of several targets of an early simulator calibration (the share of dead ball inside long gaps).
  We re-derived that parameter from the tracking alone (players' speed inside gaps vs. with ball): the
  tracking-only target selects the same value or a larger one, and the bench barely changes across that
  range (0.716 vs. 0.733 dead share inside long gaps). The final bench no longer reproduces the public
  window's figure (56 %).

None of these chose between model versions.

**Labelled data register.**

| data | labels | who | amount |
|---|---|---|---|
| Metrica sample matches | derived from their public events | algorithm | 3 matches |
| DFL/IDSSE open data | ball-status and possession flags of their tracking | provider | 7 matches |
| shape check on the challenge | aggregates of our own prediction | algorithm | — |
| **hand labels** | **none, on any match** | — | — |

## 7. Compliance with the organisers' clarification of 3 Nov

The clarification forbids training, tuning or calibrating the algorithm with the challenge's possession
ground truth or with the challenge's video, comparing outputs with the video, hand labels, and uploading
challenge data to external AI services. It allows using the video, tracking, Smart Tagging and derived
data as inputs, automatic checks, and any dataset with usage rights.

- **What we changed.** Until 10 Nov our best version used a ball detector fine-tuned on crops of the
  challenge's tactical video (pseudo-labels from the tracking, no hand labels). That is training on the
  challenge video, so it was withdrawn (section 5). The submitted model uses no video.
- **What remains and why it complies.**

| piece | what it uses from the challenge | reading |
|---|---|---|
| simulator of the bench | aggregates of the challenge's *tracking* (share of missing ball, gap lengths, ball dynamics, players' speed in gaps) to make the **external** matches look like it | no ground truth or video; no model parameter is fitted on the challenge |
| per-match features (touch threshold, team blocks as z-scores per half) | the challenge's own tracking, at inference | a fixed computation applied to the input |
| shape check (section 2.3) | aggregates of our own prediction | automatic pass/fail check, no parameter is fitted |
| short dead-ball rule (3 s), decoder costs, possession weights, possessor and its blend weight | chosen and trained on the bench only | complies |
| kick-off rule | Smart Tagging `Kick Off (Start)` and the tracked ball, as inputs | complies; the FAQ kick-off frames were only used to report its error |
| AI assistants | coding agents never read challenge data; they ran scripts that print aggregates only | no challenge data uploaded |

## 8. Limitations

- Bench labels come from events (Metrica) or from another provider's flags (DFL), not from Metrica's
  corrected tracking. The controlled-possession rule approximates the official definition but is not
  identical to it.
- Undetected steals remain the main loss, especially short possessions with no pass.
- Inside long ball gaps, accuracy drops sharply; the bench still under-represents live play inside
  crowded gaps. On the challenge the submitted model changes team directly 1.44 times per minute (bench
  ground truth 2.03), and only 0.16 per minute inside ball gaps (bench 0.98): inside a gap it tends to
  go through a dead ball instead. We accepted this risk because the bench, our only measure of accuracy,
  improves by +0.027 over the previous submission.
- The possessor network's contribution depends on its training seed: with two other seeds the bench
  gain is +0.012 instead of +0.016 (averaging three seeds: +0.015). The bench score of the submitted
  blend is therefore slightly optimistic.
- Without the video, the model cannot see who carries the ball inside the tracker's gaps. A ball
  detector trained only on external video would bring that signal back in a compliant way; there was no
  time to train and validate it before the deadline.
- With more time: a simulator that opens gaps where the tracker really loses the ball and lets dead
  balls move, a learned player-possessor model trained on many more matches with the same label
  definition, and nested validation of every choice.

## 9. Reproduction

```bash
git clone --recursive https://github.com/ManelCM/FCBCN_HACK_2027 && cd FCBCN_HACK_2027
uv sync                                  # Python 3.11, versions pinned in uv.lock
# data bundle in data/ (or MTAH_DATA_DIR pointing to it)
uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"
```

The trained models are committed in `src/hackbarca/models/artifacts/`. They can be retrained from open
data with:

```bash
uv run python -m hackbarca.pipeline.train_l3_final --bench v3_1 --no-tags --stack \
  --features touch+ball+players+blocks+structure+restart --extras transition --spell-weight \
  --temperature 1.0224660109323296 --switch 35.81318288460761 \
  --to-dead 165.64765077035295 --from-dead 10.960713295319051 --possessor-weight 0.3
uv run python -m hackbarca.pipeline.train_possessor_final     # GPU if available, ~3 min
```

The short dead-ball rule and the kick-off rule are on by default. LightGBM runs with
`deterministic=True` and forced column-wise histograms, the model files are written byte for byte
identically, and `make_submission` prints the sha256 of the file. The possessor is trained on a GPU with
a fixed seed, which does not guarantee identical bits on retraining, so its weights are versioned; the
submission runs it on CPU with deterministic algorithms. Three runs of the submitted model give the same
file, sha256 prefix `2223250da5edc7ed` (260 possessions, mean 13.6 s), and it passes the official
`validate_submission.py`. All runs were on the same machine (Windows 11, CPU).
