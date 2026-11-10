# La MasIA — Possession from automatic tracking (More Than A Hack 2027, Initial Challenge)

> DRAFT (2026-11-10). Values marked **TODO** are filled in when the model is frozen (14 Nov). Two
> variants are kept ready until then: **A** (tracking + tactical video) and **B** (tracking only). The
> one we submit is the one whose file we have reproduced bit for bit on a second machine (section 8).
> To be exported as `La MasIA.pdf`.

## 1. Approach in one paragraph

For every tactical video frame we predict whether Barcelona (H) or Athletic (A) has possession or the
ball is dead (D). spellF1 scores whole possessions with tight boundaries, so the system is built around
sequences, not frames:

1. **Per-frame features** from the automatic tracking (and, in variant A, from the tactical video):
   who is near the ball and for how long, ball touches, team shape and collective speed, restart
   layouts, and context around the gaps where the ball is missing.
2. **A stacked LightGBM**. A first layer predicts H/A/D per frame. A second layer re-reads the first
   layer's out-of-fold probabilities summarised 1-10 s backwards, centred and forwards, so it learns
   how far to trust them.
3. **An offline Viterbi decoder** per half, with switch costs between H, A and D. It turns
   probabilities into possessions and places each boundary where the evidence on both sides is best
   explained, so there is no systematic lag.
4. **Two post-processing rules**: a predicted dead ball shorter than 3 s between two live spells is
   given to the previous spell (section 3.4), and everything before each half's kick-off is D
   (section 4).

**Why this rather than something else.** There is one unlabeled match and no leaderboard, so the
decisive problem is *validation*, not model capacity. We built an external bench (section 2), fixed a
selection protocol before running experiments, and accepted a change only when it improved
consistently across held-out matches. More than 90 experiments are logged; most of them were rejected,
and section 5 lists what was tried and dropped.

## 2. Validation without a leaderboard

### 2.1 Bench
- **Ten open matches**: 3 Metrica Sports sample matches (labels derived from their events) and 7
  DFL/IDSSE Bundesliga matches (labels from the **ball-status and ball-possession flags of their
  tracking**, the same nature as the official ground truth).
- **Degraded to look like the challenge's automatic tracking**: long ball gaps (lognormal, more likely
  around dead balls), a filtered ball trajectory, tracks fragmented into 46 anonymous slots, and
  position noise.
- **Calibration targets** (aggregates only, computed locally):
  - the official baseline reproduces the eight published numbers of EVALUATION.md and their ranking;
  - the real tracking's share of graded frames without ball (40.7 %), the gap-length distribution and
    ball-dynamics percentiles (residual, velocity changes, touches per second);
  - **bench v3.1** (our final bench) also matches the rate of *slow live ball*. The first simulator
    added correlated noise that made a still ball look like it was moving at ~4 m/s, so on the bench a
    slow ball always meant a dead ball. The model learned that shortcut and produced about six times
    more short dead-ball spells on the challenge than on the bench. Fixing it moved slow live-ball runs
    from 0.006 to 0.96 per minute (clean tracking: 1.08) and made the bench show the defect.

### 2.2 Controlled possession
Event and tracking flags change team on every touch, while CHALLENGE.md only counts a change when the
new team *controls* the ball. The bench is therefore scored with a rule: a steal counts only if the
new team keeps the ball for at least T seconds (after a dead ball it always counts). T = 1.5 s for
Metrica and 2 s for DFL. With these values the bench's possession-duration distribution has the shape
of the public ground truth (17.6 % of possessions under 3 s, 14.2 s mean). T was set to understand
what "controlled" means, never by score, and it is a property of the bench labels, not of the model.
Section 6 lists this use of the public window.

### 2.3 Selection protocol (fixed before the experiments)
- Leave-one-match-out over the ten matches, pooled spellF1.
- A change is accepted only if it improves the pooled score by **≥ 0.01** *and* improves **at least 7
  of the 10** held-out matches.
- **Shape guardrail on the challenge**: a candidate that passes the bench must not make the shape of
  our own prediction on the real match worse than the current one (short dead-ball spells per minute,
  dead-ball spells per minute, direct H↔A changes per minute, share of D). These are aggregates of our
  prediction, with no ground truth.
- **Domain check**: before accepting a feature family we compare its percentiles on the real match
  and on the bench. This caught a fixed-threshold touch detector that worked on the bench and not on
  the heavily filtered real ball.
- Signals that only exist on the challenge (the video) are checked against **pseudo-labels**: our own
  confident predictions (max probability ≥ 0.9), never ground truth. They are then simulated on the
  bench with the coverage and accuracy measured on the challenge, so the bench can judge them.

### 2.4 Progression on bench v3.1 (leave-one-match-out, pooled spellF1)
| model | spellF1 | matches improved |
|---|---|---|
| model trained on the previous bench (v3), decoded as submitted | 0.7010 | — |
| retrained on bench v3.1 (variant B) | 0.7254 | 10 / 10 vs. the line above |
| + video signal: who carries the ball (variant A) | 0.7692 | 10 / 10 |
| + video signal, pessimistic simulation (stress test) | 0.7605 | 10 / 10 |
| TODO: + sustained-control features (E099 / E099nv) | TODO | TODO |

For reference, the official baseline scores 0.2433 on the public window and our first rule-based
model scored ~0.36 on the original bench. The bench is still somewhat easier than the challenge, so
we expect a lower score on the real match.

## 3. The model

### 3.1 Features (about 200 per frame)
Six families, all computed per frame from the tracking, in home/away pairs so that the model can be
trained on every match twice with the teams swapped:
- **touch**: abrupt changes in ball velocity with a per-match threshold, who touched last and who
  touches next, time since and until a touch;
- **ball**: progression before and after, long flights, still ball;
- **players**: distance of the closest player of each team, who is closest to where the ball is
  going, collective speed;
- **blocks**: length, width, area and lines of each team, relative to the match;
- **structure**: players behind the ball, in the opponent's half, shape opening or closing;
- **restart**: dead-ball layouts (players still, crowded box, player on the touchline, goalkeeper in
  the six-yard box, kick-off formation).

Every family is described with its change *after minus before* over 1-3 s windows: after a steal, the
change is informative while the current level is not.

TODO if E099 passes: **sustained control** (40 columns): fraction of the last and next 1-3 s with a
player of the team clearly closest to the ball, team support around the ball, and collective
deceleration and directional coherence.

### 3.2 Layers and training
- First layer: LightGBM, 3 classes, 127 leaves, 400 rounds, deterministic mode.
- Second layer: LightGBM on the first layer's out-of-fold probabilities with 1-10 s temporal context,
  plus the original features.
- Trained on the ten bench matches with their raw labels. The controlled-possession rule is used to
  score, not to train: training on it did not help.

### 3.3 Decoder
Viterbi per half with fixed costs (temperature 1, switch 20, to-dead 10, from-dead 10), chosen on the
bench. Tuning the decoder, a duration model (HSMM) or a minimum duration was worth +0.001 at most.

### 3.4 Short dead-ball rule
On the challenge our prediction had about 0.68 dead-ball spells under 5 s per minute; the bench ground
truth has 0.09. Most of them split one possession in two (H D H), which costs two possessions in
spellF1. The rule (a D under 3 s between two live spells takes the previous label) was fixed on the
bench, where it is neutral on clean predictions (+0.001) and recovers +0.023 when the challenge's
excess is injected. On the challenge it brings short dead-ball spells from 0.53 to 0.20 per minute.

### 3.5 Video signal (variant A only)
The tactical video sees the ball in many frames where the automatic tracking loses it, and above all
it sees *who carries it*.
- A YOLOv8n ball detector is fine-tuned on **pseudo-labels**: crops of the tactical video where the
  automatic tracking has the ball, projected with the official homography (5,974 crops with ball and
  1,456 without). No human labels.
- A COCO-pretrained YOLOv8s detects people every 5 frames. Their team comes from the torso colour,
  with colour centroids learned from people matched to a tracked player (0.926 team accuracy).
- **Carrier**: the person within 1.5 m of the most confident ball candidate. Against our confident
  predictions, its implied accuracy is ~0.90, with 40-57 % coverage depending on the context.
- Features: each team's share of carrier frames in 1, 4 and 10 s windows, coverage and its change.
  On the bench this signal is simulated with the coverage and accuracy measured on the challenge.

## 4. Possession rules implemented

- **Who owns the ball** is learned, not hand-coded, from proximity, touches, who receives, team shape
  and (variant A) the visual carrier.
- **Loose balls, deflections, duels.** Training labels never change team on a challenge, and the
  bench is scored with controlled possession. The decoder's switch cost then requires sustained
  evidence for the other team, so a deflection or a duel without control does not flip possession.
- **Restarts and dead ball.** D comes from collective speed, restart layouts, the ball near or beyond
  the lines, and the context of long ball gaps. Short dead balls inside play follow section 3.4.
- **Ball missing (41 % of graded frames).** Inside a gap the model relies on team context: who had
  the ball before and after the gap, team movement and collective speed (plus, in variant A, the
  visual carrier).
- **Before each kick-off.** The second half is graded from about two minutes before its kick-off, and
  those frames are D. The model alone predicted only 46 % D there. A general rule, with no
  match-specific constant, now sets everything before each half's kick-off to D. The kick-off comes
  from the Smart Tagging instance `Kick Off (Start)`, a ~5 s window that contains it. Inside that
  window, the kick-off is the first frame where the tracked ball moves more than 1 m. Compared with
  the kick-off frames published in the FAQ, this detection is 0.16 s late in the first half and 3.2 s
  early in the second.
- **No X on graded frames.** X appears only outside the two halves.

## 5. Assets used and dropped

| asset or idea | status | evidence (bench v3.1 unless stated) |
|---|---|---|
| tactical tracking (players + ball) | **used** | main input |
| Metrica and DFL/IDSSE open data | **used** | validation and training bench |
| tactical video + homography | **used in variant A** | visual carrier, +0.044, 10/10 |
| Smart Tagging | **used only for the kick-off rule** (section 4) | as model features it gave +0.008 (noise), and only 13 % of long gaps end with a tagged restart |
| broadcast tracking as a ball source | dropped | fills 7.5 % of the tactical gaps: it loses the ball in the same plays |
| generic ball detector (YOLO COCO) | dropped | ball within 2 m on 24 % of frames |
| filling the ball with the fine-tuned detector | dropped | +0.008, 6/10; on the real match it turned stoppages into live play |
| broadcast shot type (wide vs. cut) | dropped | 0.7311 vs. 0.7254, 6/10 |
| broadcast ball-action spotting (SoccerNet T-DEED, pretrained) | dropped | +0.005, 6/10 |
| broadcast audio: whistle, commentary, crowd, kick sound | dropped | whistle heard at only ~10 % of stoppages; other cues AUC 0.47-0.55 |
| pose (throw-ins, goalkeeper with the ball), referee position | dropped | did not pass their gates |
| ball touches from the video ball | dropped | too noisy (lift 1.38) |
| coach-style families: steal signatures, formation, game situations, pressing, territory, game state | dropped | steal signatures +0.016 alone, +0.002 on top of the video; the rest below the gate |
| player-possessor model (PathCRF-like), MS-TCN, XGBoost blend, bagging | dropped | 0.7236 / 0.7207 vs. 0.7254; blend −0.008; bagging −0.011 |
| 20 SkillCorner open matches as extra training data | dropped | 0.6856 vs. 0.7254 (worse in 9/10) |

## 6. Public window and labelled data

**Public 5-minute window.** Following the rules clarified on 22-23 Oct, it is used only as a smoke
test (format, nothing broken). It never selects, tunes or gates anything. Earlier uses, all
documented in our decision log (ADR-0019):
- the possession-duration shape behind T (section 2.2);
- a second criterion for a broadcast signal that was later dropped;
- one of several targets of an early simulator calibration, inherited by the later benches.

None of these chose between model versions.

**Labelled data register.**
| data | labels | who | amount |
|---|---|---|---|
| Metrica sample matches | derived from their public events | algorithm | 3 matches |
| DFL/IDSSE open data | ball-status and possession flags of their tracking | provider | 7 matches |
| ball detector training (variant A) | ball position from the automatic tracking, projected with the homography | algorithm (pseudo-labels) | 5,974 + 1,456 crops |
| gates on the challenge | our own confident predictions | algorithm (pseudo-labels) | — |
| **hand labels** | **none, on any match** | — | — |

## 7. Limitations

- Bench labels come from events (Metrica) or from another provider's flags (DFL), not from Metrica's
  corrected tracking. The controlled-possession rule approximates the official definition but is not
  identical to it.
- Undetected steals remain the main loss, especially short possessions with no pass.
- Inside long ball gaps, accuracy drops sharply; the bench still under-represents live play inside
  crowded gaps.
- The video signal is simulated on the bench. Its accuracy on the challenge is estimated against
  pseudo-labels, assuming errors independent of the model's.
- With more time: a simulator that opens gaps where the tracker really loses the ball, a learned
  player-possessor model trained on many more matches with the same label definition, and nested
  validation of every choice.

## 8. Reproduction

```bash
git clone --recursive https://github.com/ManelCM/FCBCN_HACK_2027 && cd FCBCN_HACK_2027
uv sync                                  # Python 3.11, versions pinned in uv.lock
# data bundle in data/ (or MTAH_DATA_DIR pointing to it)
```

**Variant B (tracking only).** The trained model is committed in `src/hackbarca/models/artifacts/`.
```bash
uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"
```

**Variant A (tracking + video, needs an NVIDIA GPU).**
```bash
uv sync --group vision
uv run --group vision python -m hackbarca.vision.pseudo_labels build   # pseudo-labelled crops
uv run --group vision python -m hackbarca.vision.pseudo_labels train   # YOLOv8n fine-tune, fixed seed
uv run --group vision python -m hackbarca.vision.video_pass            # ~8 h: ball + people per frame
uv run python -m hackbarca.pipeline.make_submission --model l3 --team-name "La MasIA"
```

The model can be retrained from open data with `hackbarca.pipeline.train_l3_final`. LightGBM runs
with `deterministic=True` and forced column-wise histograms, the model files are written byte for
byte identically, and `make_submission` prints the sha256 of the file. TODO: final variant, sha256,
commit and the second-machine check.
