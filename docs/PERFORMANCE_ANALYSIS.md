# Performance analysis: backends × decision modes

Analysis date: **2026-10-31**. Read-only analysis of the eight retained archives in
[`saved_sessions/`](../saved_sessions/README.md). No source, configuration, archive or
metric was changed. No API call or simulator run was made. The in-progress
`20261031-deepseek-v4.1-flash-1017-openrouter-316ep` run is **excluded**.
This is a diagnosis, not an approved change: any follow-up that alters navigation,
images, rejection handling or the solver needs separate approval ([AGENTS.md](../AGENTS.md)).

For a critical review of the equilibrium method itself (Consensus Game and piKL, and how the
code differs from the paper), see [EQUILIBRIUM_REVIEW.md](EQUILIBRIUM_REVIEW.md).

## 1. Summary

1. **Exploration is not what separates the runs.** All seven VLM runs offer at least one
   candidate in 86–89% of episodes, with a median first search at agent step 60 in every run.
   About 10–13% of episodes never produce a VLM candidate. That share is the same in every
   mode and backend, and only 13 episodes fail this way in all six full runs.
2. **Most losses happen in exploitation**, in three distinct ways:
   - **Wrong picks (false positives):** this is the dominant failure for DeepSeek
     Generator/Ranking, with about 125 episodes per run stopping far from the goal.
   - **False rejections:** the agent rejects every candidate and times out. This dominates
     for Jev, with 102 episodes in Jev Game.
   - **Evidence that cannot identify the target:** blind visual labels show the candidate
     composites often cannot confirm the target (§5).
3. **Equilibrium search (Game) does not help success rate.**
   - DeepSeek: Game, Ranking and Generator succeed on 101, 102 and 102 episodes respectively
     (McNemar p = 1.0).
   - Jev: Game is significantly worse than Generator (−8.5 pp, p = 0.002) and Ranking
     (−6.0 pp, p = 0.025).
   - In practice Game acts as a **Discriminator veto** on the Generator's choice. In 94–97% of
     its disagreements with the Generator's top choice, it turns a candidate pick into a reject.
   - For DeepSeek, the veto trades wrong picks for false rejections roughly one for one.
4. **Problems with the input numbers explain Jev's collapse.**
   - With one candidate, the solver selects only when g + d > 1, and Jev's Discriminator gives
     a match probability ≥ 0.5 for just **1.6%** of candidates.
   - Jev's Discriminator also barely separates acceptable candidates from unacceptable ones
     (AUC 0.58; means 0.25 vs 0.22).
   - Jev's Choice question rejects on split votes: in 22–34% of its rejections the candidates
     together had more probability than reject.
5. **Run-to-run noise is about as large as the differences between configurations.**
   - Only 24 of 316 episodes are solved by all six full runs, while 184 are solved by at
     least one.
   - DeepSeek Generator and Ranking have identical success rates but share only 72 of their
     102 successes.
   - At 316 episodes, the 95% confidence interval on success rate is about ±5 pp.

## 2. Data and method

| Run key | Archive | Episodes | SR | SPL |
|---|---|---:|---:|---:|
| clip | 20261025-clip-baseline-316ep | 316 | 0.247 | 0.094 |
| ds_gen / ds_rank / ds_game | DeepSeek V4.1 Flash (direct API) Generator / Ranking / Game | 316 each | 0.323 / 0.323 / 0.320 | 0.116 / 0.122 / 0.121 |
| mini_game | GPT-4o-mini Game (OpenRouter; stopped on key limit) | 188 | 0.293 | 0.104 |
| jev_gen / jev_rank / jev_game | Jev 1.14 Decisions Generator / Ranking / Game | 316 each | 0.301 / 0.275 / 0.215 | 0.081 / 0.074 / 0.064 |

- **Sources:**
  - For each VLM run: the `episode_start/end`, `search_*`, `generator_aggregate`,
    `discriminator_aggregate` and `policy_snapshot` events in every `equilibrium.jsonl`.
  - For CLIP: the per-episode lines of its run-only log. Its per-episode built-in category
    comes from the difference between consecutive running totals.
- **Checks:**
  - Recomputed SR, SPL and DTG match every session README.
  - The built-in failure counts match the logs.
  - mini's incomplete episode 189 is excluded.
- **Episode alignment:** every run follows the same episode order (316/316 dataset IDs match).
  Perception is not deterministic across runs: only about 25% of first searches have the same
  step and candidate IDs. So the comparisons below pair runs by episode, not by candidate.
- **Solver replication:** an offline copy of the equilibrium solver reproduces all 2,132
  archived solved Game searches exactly (decision parity 2,132/2,132; maximum π_G difference
  9e-16).
- **Visual labels:** 250 searches (690 candidate composites) were labelled blind. Labellers saw
  only the instruction, the candidate IDs and the exact archived JPEGs, with no information
  about the model's decision or the episode outcome.
  - **Sample:** stratified by run and decision (Game: 18 flips, 16 rejections and 16
    selections per run; other modes: 12 rejections and 13 selections per run), limited to
    searches with ≤ 8 candidates. Run-level rates are weighted back to each run's population.
  - **Labellers:** done by Claude (five annotator passes under one rubric). A 30-search
    overlap gave per-candidate agreement of 0.90, κ = 0.79, and best-choice agreement of 0.97.
  - **Acceptable (lenient):** the category is "yes" or "likely" and the spatial relation is
    not contradicted.
  - **Strict:** the category is "yes" and the relation is consistent or partial.
  - Labels measure whether the image plausibly shows the described object. They do not show
    which object instance the dataset counts as the goal, because no geometric ground truth
    is logged and simulator replay was ruled out of scope.

## 3. Exploration vs exploitation

### 3.1 Refined outcome taxonomy

The code's built-in `exploration` label means only "reached 500 steps" (`main_vln.py:206-213`),
so an episode that rejects every candidate until it runs out of steps is counted as an
exploration failure. The refined categories are:

| Code | Meaning |
|---|---|
| S-vlm / S-fallback | Success after a VLM selection / success through the CLIP fallback stop (`no_frontiers_count > 5`) |
| E0 | No VLM candidate was ever offered (timeout, or fallback stop) → exploration/perception |
| E1 | Candidates were offered and **all were rejected** (timeout, or fallback stop) → false rejection or a correct candidate never offered |
| X1-near / X1-far | Selected a candidate and stopped ≤ 1 m / > 1 m from a goal view point (success radius 0.25 m) |
| X2 | Selected a candidate, then timed out |

| Category | clip | ds_gen | ds_rank | ds_game | mini (188) | jev_gen | jev_rank | jev_game |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Success (VLM / fallback) | 78 | 91 / 11 | 95 / 7 | 92 / 9 | 50 / 5 | 76 / 19 | 67 / 20 | 44 / 24 |
| E0 no candidate (timeout / stop) | – | 14 / 16 | 26 / 14 | 14 / 16 | 12 / 9 | 24 / 14 | 12 / 14 | 18 / 20 |
| E1 all rejected (timeout / stop) | – | 16 / 15 | 15 / 17 | **53** / 28 | 24 / 12 | 53 / 25 | 57 / 28 | **102** / 36 |
| X1 wrong stop (near / far) | 34 / 180 | 23 / **125** | 18 / **123** | 25 / 73 | 18 / 54 | 21 / 80 | 20 / 93 | 14 / 57 |
| X2 selected then timeout | – | 5 | 1 | 6 | 4 | 4 | 5 | 1 |
| CLIP timeout | 24 | – | – | – | – | – | – | – |

### 3.2 Candidate supply and selection behaviour

| | ds_gen | ds_rank | ds_game | mini | jev_gen | jev_rank | jev_game |
|---|---:|---:|---:|---:|---:|---:|---:|
| Episodes with ≥ 1 candidate | 0.886 | 0.861 | 0.883 | 0.878 | 0.864 | 0.889 | 0.870 |
| Search reject rate | 0.546 | 0.549 | 0.773 | 0.752 | 0.800 | 0.809 | **0.905** |
| Episodes with a selection | 0.772 | 0.750 | 0.620 | 0.670 | 0.573 | 0.585 | **0.367** |
| SR given a selection | 0.373 | 0.401 | 0.469 | 0.397 | 0.420 | 0.362 | 0.379 |
| Median step of selection | 100 | 100 | 120 | 120 | 160 | 160 | 140 |
| SPL on successful episodes | 0.36 | 0.38 | 0.38 | 0.36 | 0.27 | 0.27 | 0.30 |

- A selection is never reconsidered. After a selection the agent stops a median of 5–7 steps
  later, so the outcome of a selection is almost entirely the outcome of the choice.
- Game raises DeepSeek's precision when it does select (0.47 vs 0.37–0.40) but selects in
  fewer episodes (0.62 vs 0.75–0.77), so success rate stays flat.
- For Jev, Game does not raise precision (0.38 vs Generator 0.42); it only selects less often.
- Rejections cost steps and SPL.
  - Episodes that reject everything and time out use **48%** of Jev Game's 106,102 agent steps
    (51,000) and 32% of DeepSeek Game's (26,500).
  - Later selections lower Jev's SPL even on successful episodes (0.27–0.30 vs 0.36–0.38).

### 3.3 Episodes that are always hard

Across the six full runs, 134 episodes are never solved. Within those episodes:

- 65 have a wrong pick (X) in ≥ 4 runs.
- 44 have all-rejected (E1) in ≥ 3 runs.
- 25 have no candidate (E0) in ≥ 3 runs.

Wrong selections dominate this hard core. Some targets do badly across all backends, e.g.
`cabinet` (63 episodes, 0.11–0.27 SR) and `bed` (VLMs 0.0–0.1 vs CLIP 0.4).

## 4. Mode and backend comparison (paired by episode)

| A vs B | SR A | SR B | A only | B only | McNemar p | ΔSR 95% CI |
|---|---:|---:|---:|---:|---:|---|
| ds_game vs ds_gen | 0.320 | 0.323 | 34 | 35 | 1.00 | [−0.054, 0.047] |
| ds_game vs ds_rank | 0.320 | 0.323 | 36 | 37 | 1.00 | [−0.057, 0.047] |
| ds_rank vs ds_gen | 0.323 | 0.323 | 30 | 30 | 1.00 | [−0.047, 0.047] |
| jev_game vs jev_gen | 0.215 | 0.301 | 22 | 49 | **0.002** | [−0.136, −0.035] |
| jev_game vs jev_rank | 0.215 | 0.275 | 23 | 42 | **0.025** | [−0.108, −0.009] |
| jev_rank vs jev_gen | 0.275 | 0.301 | 28 | 36 | 0.38 | [−0.076, 0.025] |
| ds_game vs jev_game | 0.320 | 0.215 | 55 | 22 | **<0.001** | [0.051, 0.158] |
| ds_gen vs jev_gen | 0.323 | 0.301 | 44 | 37 | 0.51 | [−0.035, 0.079] |
| ds_game vs mini (188) | 0.319 | 0.293 | 23 | 18 | 0.53 | [−0.043, 0.096] |
| ds_game vs clip | 0.320 | 0.247 | 53 | 30 | **0.015** | [0.016, 0.127] |
| jev_gen vs clip | 0.301 | 0.247 | 51 | 34 | 0.08 | [−0.003, 0.111] |
| jev_game vs clip | 0.215 | 0.247 | 36 | 46 | 0.32 | [−0.089, 0.025] |

- DeepSeek beats CLIP in every mode.
- Jev Generator is not statistically different from DeepSeek Generator.
- Jev Game is below CLIP in point estimate (not significant).
- GPT-4o-mini is not distinguishable from DeepSeek on its 188 episodes.

## 5. Visual-label results (exploitation quality)

Rates are population-weighted, over searches with ≤ 8 candidates, using lenient labels.

| | ds_gen | ds_rank | ds_game | mini | jev_gen | jev_rank | jev_game |
|---|---:|---:|---:|---:|---:|---:|---:|
| Searches with an acceptable candidate offered | 0.61 | 0.52 | 0.51 | 0.43 | 0.66 | 0.42 | 0.66 |
| **False-rejection rate** (rejected, but acceptable candidate offered) | 0.42 | 0.25 | 0.42 | 0.26 | 0.58 | 0.33 | **0.63** |
| Selection precision (selected candidate acceptable) | 0.84 | 0.85 | 0.81 | 0.81 | 0.85 | 0.53 | 0.79 |
| Searches where no candidate can be judged | 0.18 | 0.21 | 0.21 | 0.32 | 0.26 | 0.15 | 0.12 |

Each per-run cell rests on 25–50 labelled searches, so treat single-run differences as
indicative. The pooled figures are more reliable:

- Rejected searches with an acceptable candidate offered:
  - chat models 40/91 = 0.44 [0.34, 0.55]
  - Jev 32/57 = 0.56 [0.42, 0.69]
- Failed selections whose selected candidate was acceptable:
  - chat models 26/33 = 0.79 [0.61, 0.91]
  - Jev 15/25 = 0.60 [0.39, 0.79]

Interpretation:

- Most **failed** selections were visually plausible picks of the right category. The final
  failure then comes from **which instance** was picked, or from where the agent stopped
  (X1-near is 14–25 episodes per run), rather than from an obviously wrong category. A third of
  failed selections had several plausible instances.
- **The candidate images are weak evidence.**
  - Across all labelled candidates, only 8% are a clear "yes" and 32% are "cannot tell".
  - Relations are "unknown" for 50%.
  - The first-view frame shows the target category at all in only 52% of composites.
  - Image quality is good for 57%, poor for 40% and unreadable for 3%.
  - Acceptable rate falls from 0.53 for good images to 0.26 for poor and 0 for unreadable.
  - Even among successful selections, only 34% meet the strict standard.
- The cause is in how the composite is built. The first-view half is the frame where the
  object was **first detected** (`image_idx[0]`); it is not centred on the object and does not
  mark it. The only cue to which object is meant is a small red box on the top-down point
  cloud. A cautious judge therefore often cannot confirm the target, and rejecting is a
  reasonable answer.

## 6. Does the equilibrium search help?

### 6.1 Where Game makes its decisions

| | ds_game | mini | jev_game |
|---|---:|---:|---:|
| Early reject (all Generator votes were reject; solver never ran) | 26.5% | 45.1% | 0% (Jev keeps every option) |
| Solver rejected | 50.8% | 30.1% | 90.5% |
| Solver selected | 22.7% | 24.8% | 9.5% |
| Game = Discriminator top choice (solved searches) | 95.4% | 86.7% | 94.3% |
| Game = Generator top choice (solved searches) | 66.0% | 63.8% | 86.1% |
| Disagreements with the Generator that turn a candidate pick into a reject | 203/216 | 97/101 | 160/169 |
| Generator candidate picks vetoed (solved searches) | 51.8% | 43.5% | 58.2% |

### 6.2 What the solver does with its inputs

Enumerating every input pattern on the verified solver gives these rules:

- **One candidate (Jev):** select if and only if d ≥ 1 − g, i.e. **g + d ≥ 1**.
- **One candidate (chat counts, Generator c/5 and Discriminator k/5):**

  | Generator c/5 | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | Discriminator k/5 needed to select | ≥ 4 | ≥ 3 | ≥ 3 | ≥ 2 | ≥ 1 |

  This is a threshold on the average of the two probabilities; ties at c + k = 5 go to
  reject when c ≥ 3.
- **Two candidates (2,160 chat patterns):** 79% of patterns select a candidate. 72% of
  decisions match "highest Discriminator score unless the reject-all score is at least as
  high"; most of the rest reject when the Generator supports neither candidate strongly.
- After 500 iterations the accumulated payoff (≈ ½·H₅₀₀·d ≈ 3.4·d) outweighs the Generator
  prior term (0.1·log g). The equilibrium therefore follows the Discriminator, while the
  Generator **gates** which candidates are considered:
  - chat code drops candidates with zero Generator votes;
  - an all-reject vote skips the solver.
- The game-theoretic machinery does not add information beyond an average or veto rule over
  the two models' scores. Its result is only as good as those scores.

### 6.3 Were the vetoes right?

- Generator picks that Game vetoed were visually acceptable in **31/52 = 0.60 [0.45, 0.73]**
  of cases. Picks that Game kept were acceptable in 0.81 [0.67, 0.91]. On strict labels the
  figures are 0.12 vs 0.27.
- The veto does aim at somewhat weaker candidates, but it also removes many plausible ones.
- Only 2 of the 26 non-veto flips were labelled; in both, Game's choice was acceptable.
- Outcome of episodes containing ≥ 1 veto:

  | | Later success | All rejected | Wrong stop | Selected, then timeout |
  |---|---:|---:|---:|---:|
  | ds_game | 20 | 38 | 30 | 5 |
  | jev_game | 17 | 34 | 18 | 1 |
  | mini | 10 | 17 | 19 | 2 |

**Verdict:**

- DeepSeek: the equilibrium makes no net contribution. Its higher precision is cancelled by
  lost coverage.
- Jev: it is harmful, because the Discriminator signal it follows is both biased low and
  barely informative.
- No variant improves on the plain Generator for either backend.

## 7. Are the input numbers the problem?

Yes. The solver works only on the numbers it receives, and several properties of those numbers
determine the outcome.

1. **Jev's Discriminator (Noul) is compressed and uninformative.**
   - Candidate match probabilities have median 0.196, 95th percentile 0.41 and maximum 0.80.
     Only 1.6% are ≥ 0.5.
   - Against the visual labels, its AUC is **0.58** (mean 0.25 for acceptable candidates vs
     0.22 for others).
   - On outcome-labelled selections its AUC is **0.41**, worse than chance.
   - The question asks whether the candidate matches "the described target, including **all
     required** attributes and spatial relationships". The auto-generated instructions often
     have loose or odd relations, and the images rarely show them (§5). So a conjunctive yes/no
     is almost never likely, and under g + d ≥ 1 nearly everything is rejected.
2. **DeepSeek's Discriminator is informative but strict.**
   - AUC 0.74, about equal to its Generator (0.73).
   - It is bimodal: 64% of candidates get 0/5 "correct" and 16% get 5/5.
   - So it rejects many plausible candidates (§6.3).
   - GPT-4o-mini's Discriminator is weaker (AUC 0.61).
3. **Jev's Choice distribution produces split-vote rejection.**
   - Its mean reject probability (0.52–0.56) is similar to the chat models'.
   - But reject is one option competing against several candidates that share the rest.
   - In 22–34% of Jev's rejections the candidates together outweighed reject (chat models:
     0–1.7%).
   - Jev's Generator scores still rank candidates reasonably (AUC 0.62–0.74).
4. **Chat votes are coarse and often unanimous.**
   - Probabilities are k/5 plus a 0.05 bias, so the solver sees a small set of patterns.
   - Even at temperature 1.1, the Generator is unanimous in 41% (DeepSeek) and 73% (mini) of
     Game searches.
   - So the "distribution" given to the solver is often just a single vote repeated.
5. **The chat reject-all score can contradict the per-candidate scores.**
   - With ≥ 2 candidates, the reject-all Discriminator score comes from a separate all-images
     query.
   - In 19% (mini) and 6.6% (DeepSeek) of multi-candidate searches, both some candidate's
     score and the reject-all score were ≥ 0.6.
6. **The evidence itself is weak** (§5): the first-view frame is uncentred and unmarked,
   40% of images are poor, and relations are unverifiable. This lowers both Generator and
   Discriminator confidence. It is the input problem that affects every backend.

## 8. Limitations

- There is no geometric ground truth for candidates. Visual labels judge category and
  plausibility, not the dataset's goal instance, and depend on rubric choices: attribute
  mismatches count as "likely", and duplicate boxes on one object were handled
  inconsistently.
- The label sample is limited to searches with ≤ 8 candidates (most searches qualify; the
  sample mean is 2.8 candidates).
- Comparing decision rules on the same solver inputs does not give episode success rates,
  because a different decision changes the rest of the trajectory. Effects on success rate
  come only from the paired comparison of whole runs.
- mini has 188 episodes. Providers differ (DeepSeek direct API; mini via OpenRouter; Jev via the
  Decisions API), so backend comparisons also compare transport and model.
- Run-to-run variation is large (Summary, item 5); one run per configuration cannot separate small effects.

## 9. Possible follow-ups (proposals only; none approved)

- Measure first-view selection (centred or marked frames, best-view instead of first-detection
  frame) in a short, separately approved experiment. This is an imagery/behaviour change.
- For Jev: test a Discriminator question that asks only about the main target and attributes,
  or compare against a rank-based rule. This changes interpretation and prompts, so it needs
  approval and must not be tuned on benchmark success rate.
- Repeat runs (≥ 2 per configuration) before drawing conclusions from differences under 5 pp.
- Optional geometric ground truth through instrumented simulator replay (out of scope for this
  analysis).
