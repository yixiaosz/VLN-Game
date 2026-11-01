# Critical review: why the equilibrium benefit does not carry over

Date: **2026-10-31**. This review follows [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)
and asks why the original VLN-Game paper reported a benefit from equilibrium search while our
runs with other foundation models do not.

- **Sources:**
  - VLN-Game: Yu et al., arXiv:2411.11609 (§III-F, Algorithm 1, Eqs. 7–15, Table III).
  - Consensus Game (CG): Jacob, Shen, Farina, Andreas, arXiv:2310.09139 / ICLR 2024.
  - piKL: Jacob et al., arXiv:2112.07544 / ICML 2022.
  - The archived traces.
- **Scope:** all experiments here are offline replays of the archived solver inputs. No source
  code, archive, API call or simulator run was involved. Everything proposed in §7 needs
  separate approval.

## 1. Short answer

1. **The paper's own evidence for equilibrium is weak.**
   - Its Table III reports Game minus Ranking as **+0.3 pp with GPT-4o** (0.367 vs 0.364, one
     episode of 316) and **+2.2 pp with GPT-4o-mini** (0.354 vs 0.332, seven episodes).
   - Each number comes from one run, with no confidence intervals or paired test.
   - In our archives, same-model configurations differ by 30–37 episodes in *each* direction,
     and the 95% interval on SR is about ±5 pp.
   - Our DeepSeek result (Game − Ranking = −0.3 pp) is therefore statistically consistent with
     the paper's GPT-4o row. The equilibrium "benefit" was never resolved from noise; we have
     not failed to reproduce an established effect.
2. **The rest of the pipeline reproduces well.** The paper's CLIP baseline is 0.253 and ours is
   0.247. Exploration and perception are not why the VLM results differ.
3. **What the code computes is not quite the published method.** The released code differs
   from both the Consensus Game and VLN-Game's own equations in ways that make the "game" a
   **Discriminator-dominated accept/reject threshold** (§4). Under it, the result depends on how
   the particular model's Discriminator is scaled and calibrated.
4. **Changing the foundation model changes the only thing the solver sees: the numbers.**
   Equilibrium search adds no information about the environment (§3). It can only help when
   the Generator and Discriminator scores are informative, on comparable scales and have
   complementary errors.
   - DeepSeek meets the first condition but its Discriminator is too strict, so Game trades
     wrong picks for false rejections and the net effect is zero.
   - Jev fails on scale (match ≥ 0.5 for only 1.6% of candidates) and on informativeness
     (AUC 0.58), so Game makes it worse.

## 2. What the original paper actually shows

| Paper Table III (316 episodes) | SR | SPL | Δ SR vs Ranking |
|---|---:|---:|---:|
| CLIP-based | 0.253 | 0.106 | |
| GPT-4o-mini: Generator / Ranking / **Game** | 0.304 / 0.332 / **0.354** | 0.108 / 0.113 / 0.118 | **+0.022** |
| GPT-4o: Generator / Ranking / **Game** | 0.351 / 0.364 / **0.367** | 0.120 / 0.124 / 0.132 | **+0.003** |

- **Detectability:**
  - In our paired DeepSeek data about 23% of episodes are discordant between two
    configurations (73/316).
  - Detecting a 2 pp SR difference with McNemar's test at α = 0.05 and 80% power needs about
    7.84 × 0.23 / 0.02² ≈ **4,500 paired episodes**, roughly fourteen times the 316-episode
    benchmark.
  - A single 316-episode run cannot show the effect sizes in Table III.
- **The paper's own interpretation:**
  - The paper's statement that "VLN-Game with GPT-4o-mini even surpasses … VLN-Generator with
    GPT-4o" is a 0.3 pp difference.
  - The claim that the gain is larger for the smaller model rests on +2.2 vs +0.3 pp; both are
    within noise.
- **Which code produced Table III:** the released code (commit `58ca7f4`) has the
  Discriminator accumulation defect corrected in change 12 (§4, row 10). Which revision
  produced Table III is unknown.
- **What we can compare:** our mini Game (0.293 on 188 episodes) is not directly comparable
  with the paper's 0.354 on 316. We have no per-episode paper results, and the provider route
  and temperature differ.

## 3. The logic of the two source methods

**piKL (KL-regularised search):**

- Each player maximises *expected game utility* − λ·KL(π ‖ anchor). The anchor is a human
  imitation policy.
- The utility comes from the real game (rollouts or value estimates in chess, Go and
  Diplomacy), so search brings **new information** (lookahead), and the KL term keeps play
  human-like.
- piKL-Hedge is a no-regret method for computing this regularised equilibrium. Its guarantees
  are about convergence to the regularised equilibrium, not about that equilibrium being
  correct.

**Consensus Game (CG):**

- Nature draws v ∈ {correct, incorrect} uniformly.
- The Generator sees v and emits a candidate y. The Discriminator sees y and guesses v.
- Both players are paid when the guess is right.
- piKL keeps both close to the LM's initial policies.
- Here the utility is only the agreement between two ways of querying the same model. The
  equilibrium therefore **cannot add information**. It reconciles the generative and
  discriminative views, acting as a regularised, consistency-seeking ensemble.
- Its demonstrated gains depend on conditions that are worth listing:
  - **Dense, comparable scores:** token log-likelihoods, cross-normalised; the Generator over
    v, the Discriminator over candidates.
  - **A genuinely informative "incorrect" channel:** a separate "Incorrect Answer:" prompt.
  - **Fixed candidate sets with no reject option:** multiple-choice options, or sampled
    candidates for TruthfulQA and GSM8K.
  - **Benchmarks of thousands of items,** so 1–3 point gains are measurable.
- CG's own margins over the best baseline are mostly +1–3 points. It is sometimes worse,
  e.g. 7B MMLU (Discriminator-only 40.4 vs ER-D 39.9) and 13B HHH unless λ is re-tuned.
- **Implication:** under KL-regularised coordination the equilibrium is effectively a fixed
  log-linear pooling of the two initial policies, roughly π_G* ∝ π_G⁽¹⁾^a · exp(b·π̄_D). The
  weights a and b are set by λ, η, T and how Q is accumulated. It is only as good as the
  numbers pooled and the balance between them.

## 4. Where the VLN-Game implementation departs from CG and from its own paper

Each departure was measured on archived inputs where possible (2,132 solved Game searches; the
solver replica matches 100%).

| # | Aspect | CG | VLN-Game paper | Archived code | Measured effect |
|---|---|---|---|---|---|
| 1 | Payoff accumulation | Q = (1/2t)·Σπ (running average, ≤ ½) | Eq. 12: same as CG | `Q += π_t/(2t)` (harmonic sum ≈ ½·H_T ≈ 3.4 at T = 500) | Weights the opponent's payoff 5–7× more against the KL anchor. Switching to the paper form lowers the reject rate: DeepSeek 0.773 → 0.728, mini 0.752 → 0.634, Jev 0.905 → 0.899 |
| 2 | Iterations | 5,000 | not stated | 500 | 0/2,132 decisions change at 5,000 |
| 3 | Generator "incorrect" policy | separate "Incorrect Answer:" prompt | not specified | 1 − g, renormalised | Carries no information beyond g (cannot be tested offline) |
| 4 | Discriminator normalisation | across candidates | Eq. 11: across candidates | per candidate over {correct, incorrect} | Applying the paper form changes reject rates by ≤ 0.005 |
| 5 | Initial scores | LM likelihoods | 5 samples, temperature 1.0, + bias | 5 samples, temperature **1.1**, + 0.05 | Chat Generator unanimous in 41% (DeepSeek) / 73% (mini) of Game searches; scores quantised to k/5 |
| 6 | Candidates with zero Generator votes | n/a (likelihoods never 0) | – | dropped from the game | Game cannot select them |
| 7 | All-reject Generator vote | n/a | – | solver skipped (early reject) | 26.5% (DeepSeek) / 45.1% (mini) of Game searches |
| 8 | Reject option | **none** | rejection described in prose | −1 is a pseudo-candidate; its Discriminator score is 1 − d (one candidate) or a separate all-image query | 72% (DeepSeek) / 83% (mini) of solved searches have **one** candidate, so the game reduces to **select iff g + d ≥ 1**. Jev: 24% single |
| 9 | Stated payoff | correctly identify v | Eq. 7: both paid when the Discriminator says "correct" (no v) | CG-style rows | The paper's game statement does not match its own update |
| 10 | Discriminator pool | per candidate | – | original `58ca7f4`: accumulated across candidates (fixed in change 12) | Emulated on our counts: changes 12% / 26% of multi-candidate decisions (~3.5–4% of all solved), NaN policies in 17–26% of them, reject rate 0.72 → 0.70. Too small to explain the gap |

**Result:** rows 1, 5, 6 and 8 together turn the "consensus game" into a coarse, cost-blind
accept/reject rule that is dominated by the Discriminator. Game matches the Discriminator's top
choice in 87–95% of solved searches, and 94–97% of its disagreements with the Generator are
vetoes.

## 5. Why the foundation model matters so much

The solver works only on the numbers it receives, so what matters is how each model's
Generator and Discriminator behave on these inputs (visual-label AUC from PERFORMANCE_ANALYSIS
§5–7):

| | DeepSeek | GPT-4o-mini | Jev 1.14 |
|---|---:|---:|---:|
| Generator AUC (acceptable vs not) | 0.73 | 0.73 | 0.69 |
| Discriminator AUC | 0.74 | 0.61 | **0.58** |
| Discriminator mean score, acceptable / other | 0.42 / 0.08 | 0.41 / 0.27 | **0.25 / 0.22** |
| Discriminator ≥ 0.5 | 23% | 31% | **1.6%** |
| corr(g, d) | 0.53 | 0.34 | 0.41 |
| Solved searches with one candidate | 72% | 83% | 24% |
| Game reject rate | 0.77 | 0.75 | **0.91** |

Reading the table:

- A pooled equilibrium helps only if the Discriminator brings **extra, correctly scaled
  evidence**.
  - DeepSeek's Discriminator is informative but harsher than its Generator, so the veto
    raises precision (0.47 vs 0.37–0.40 SR given a selection) at the cost of coverage. The net
    effect is about zero.
  - mini's Discriminator is weaker. Jev's is both compressed (max 0.80) and nearly
    uninformative, so the pooling mostly injects a reject bias.
- **The question and interface are part of the "model".**
  - The chat Discriminator is told to answer "correct if any image matches".
  - Jev's Noul asks whether the candidate matches "the described target, including **all**
    required attributes and spatial relationships". Given loose auto-generated relations and
    images that rarely show them (only 8% of candidates clear-yes, relations unknown in 50%),
    a calibrated model should give low probabilities. The threshold g + d ≥ 1 then rejects
    almost everything.
  - Jev's Choice distribution adds split-vote rejection: reject wins although the candidates
    together outweigh it in 22–34% of its rejections.
- **The paper's own design assumed sampled chat votes at temperature 1.0 from GPT-4o.** The
  equilibrium's weights (λ = η = 0.1, bias 0.05, harmonic Q) were not re-derived when the
  input statistics changed. CG itself notes that tuning λ changes results materially, e.g. on
  HHH. Re-tuning on the benchmark is out of scope under AGENTS rule 1; the point is that the
  method has no model-independent operating point.

## 6. Decision-level evidence

Accuracy of each decision rule against the blind visual judge, on the labelled Game searches
(lenient labels; weighted to population; about 50 searches per run, so ±0.13):

| Run | Archived Game (code Q) | Generator argmax | Game with paper Q (running avg) | + paper Discriminator normalisation | Discriminator argmax |
|---|---:|---:|---:|---:|---:|
| DeepSeek | 0.63 | 0.68 | 0.70 | 0.70 | 0.63 |
| GPT-4o-mini | 0.76 | **0.84** | 0.79 | 0.79 | 0.72 |
| Jev | 0.42 | **0.44** | 0.42 | 0.42 | 0.37 |

- No equilibrium variant clearly beats the plain Generator.
- The closer a rule is to "follow the Discriminator", the lower its accuracy.
- The paper-faithful running-average Q sits between Generator and code-form Game, as its
  weaker payoff weight predicts.
- These are decision-level comparisons only; episode success also depends on the trajectory.

## 7. Critical assessment and what it would take

1. **Equilibrium ranking only fuses scores.** It is not a source of new information. Unlike
   piKL in Diplomacy, there is no environment utility. Any benefit must come from G/D
   complementarity, so it is model-specific by construction.
2. **CG's guarantees do not cover the VLN setting:**
   - an open candidate set that may not contain the target;
   - an asymmetric, time-costly reject option;
   - irreversible selection (a choice is never revisited);
   - searches that re-trigger only when the candidate count changes.
   The prior P(v = correct) = ½ and the reward "Discriminator identifies v" do not encode the
   navigation cost of rejecting a correct target, or of accepting a wrong one.
3. **The released code is not the published equations** (§4 rows 1, 4, 9, 10). Results from
   the code cannot be attributed to the method as described, and vice versa.
4. **The headline comparison is underpowered.** Ranking vs Game differences of 0.3–2.2 pp need
   repeated runs or thousands of episodes (§2).
5. **Possible next steps (proposals; none approved, none to be tuned on benchmark SR):**
   - Repeat Ranking vs Game (e.g. ≥ 3 seeds each), or evaluate decisions offline on fixed
     candidate sets, before attributing effects to the equilibrium.
   - Test a CG-faithful variant: likelihood or logprob inputs where the provider exposes them,
     a separate "incorrect" Generator prompt, the paper's running-average Q and cross-candidate
     normalisation. Keep accept/reject outside the game as an explicit, documented decision
     rule.
   - For Jev: align the Noul question with the chat Discriminator's criterion, or score target
     and attributes separately from relations.
   - Improve the evidence: centred, marked first-view frames. This is an imagery change that
     needs approval.

## Appendix: reproduction notes

- **Offline solver and variant scripts:** session scratchpad `analysis/solver.py` and
  `analysis/variants.py`. They are not stored in the repo unless requested.
- **Parity:**
  - The code-form replica matches all 2,132 archived decisions.
  - The dict-arithmetic emulation of `58ca7f4` reproduces the corrected archive decisions
    exactly (100%) before the accumulation is applied.
- **Paper text:** extracted from arXiv:2411.11609 (PDF). CG details are from the arXiv HTML
  version. piKL details are from the original paper as summarised above; no numeric piKL
  claims are used.
