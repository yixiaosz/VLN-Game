# Executive summary: Stage I performance analysis

**Date:** 2026-10-31.
**Scope:** read-only analysis of the eight retained full or partial runs (316 language-goal
episodes; GPT-4o-mini 188). The in-progress 1017 OpenRouter run is excluded. No code, archive
or metric was changed.
**Details:** [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) (data and diagnosis) and
[EQUILIBRIUM_REVIEW.md](EQUILIBRIUM_REVIEW.md) (method review against the Consensus Game and
piKL).

## Results at a glance

| Backend | Generator | Ranking | Game | Reference |
|---|---:|---:|---:|---|
| DeepSeek V4.1 Flash | 0.323 | 0.323 | 0.320 | CLIP baseline 0.247 |
| Jev 1.14 | 0.301 | 0.275 | **0.215** | |
| GPT-4o-mini (188 episodes) | – | – | 0.293 | |
| *Paper, GPT-4o / GPT-4o-mini* | *0.351 / 0.304* | *0.364 / 0.332* | *0.367 / 0.354* | *paper CLIP 0.253* |

Values are success rates. The 95% confidence interval on each run's SR is about ±5 pp.

## Key findings

1. **Exploration is not the bottleneck.**
   - Every VLM run gets candidate objects in 86–89% of episodes, first at step 60.
   - About 10–13% of episodes never produce a candidate, in every mode and backend.
   - Our CLIP baseline matches the paper (0.247 vs 0.253), so mapping, exploration and
     perception reproduce.
2. **Performance is lost when choosing among the candidates, in three ways.**
   - **Wrong picks:** this dominates DeepSeek Generator and Ranking (about 125 episodes per run
     stop at the wrong object). Most of these were the right category but the wrong instance.
   - **False rejections:** this dominates Jev. In 58–63% of Jev's rejected searches a visually
     acceptable candidate was on offer, and Jev Game spends 48% of its steps in episodes that
     reject everything until they time out.
   - **Weak visual evidence:** the first-person image is the frame where the object was first
     detected, not centred and not marked. It shows the target category in only 52% of cases,
     so the VLM often cannot confirm the target.
3. **Equilibrium search (Game) does not improve success.**
   - DeepSeek: Game, Ranking and Generator are statistically identical (McNemar p = 1.0).
   - Jev: Game is significantly worse than Generator (−8.5 pp, p = 0.002) and Ranking
     (−6.0 pp, p = 0.025).
   - In practice the Game acts as a **Discriminator veto**: 94–97% of its disagreements with
     the Generator turn a pick into a reject, and 60% of vetoed picks were visually acceptable.
   - With one candidate, which is 72–83% of chat-model searches, it reduces to "accept if
     g + d ≥ 1".
4. **The input numbers decide the outcome.**
   - Jev's Discriminator gives a match probability ≥ 0.5 for only 1.6% of candidates and
     barely separates good from bad candidates (AUC 0.58).
   - Jev's Choice question also rejects on split votes: in 22–34% of its rejections the
     candidates together outweighed reject.
   - DeepSeek's Discriminator is informative (AUC 0.74) but strict: Game's precision rises
     (0.47 vs 0.37–0.40) while coverage falls, so the net effect is zero.
5. **The paper's equilibrium benefit was never statistically established.**
   - Its Game-over-Ranking gains are +0.3 pp (GPT-4o) and +2.2 pp (GPT-4o-mini), from single
     316-episode runs.
   - Detecting a 2 pp effect needs about 4,500 paired episodes.
   - Our DeepSeek result (−0.3 pp) is consistent with the paper's GPT-4o row.
6. **The released code differs from the published method.**
   - The payoff accumulates as a harmonic sum, not the paper's running average, which weights
     the Discriminator 5–7× more heavily.
   - The "incorrect" Generator channel is 1 − g rather than a separate prompt.
   - Inputs are sampled votes, not likelihoods.
   - "Reject" is added as a pseudo-candidate, which Consensus Game does not have.
   - The original Discriminator bug (fixed in change 12) and the iteration count (500 vs
     5,000) have little effect: about 4% of decisions and 0 decisions respectively.
7. **Run-to-run noise is about as large as the effects being studied.**
   - Only 24 episodes are solved by all six full runs; 184 are solved by at least one.
   - DeepSeek Generator and Ranking share only 72 of their 102 successes each.

## Bottom line

The reproduction pipeline is faithful, but neither the paper nor our data supports a benefit
from equilibrium search on this benchmark. The Game's effect depends on the model:

- it is neutral when the Discriminator is informative but strict (DeepSeek);
- it is harmful when the Discriminator is compressed and uninformative (Jev).

The main ceiling on performance is choosing among candidates on weak visual evidence, not
exploration.

## Caveats

- **Labels:** visual labels (250 searches, labelled blind by Claude under a fixed rubric,
  κ = 0.79) judge plausibility, not the dataset's goal instance. No geometric ground truth
  exists.
- **Decision-level comparisons:** these do not give episode success rates.
- **Providers:** providers and routes differ across backends.
- **GPT-4o-mini:** only 188 episodes.

## Decisions for Stage II (proposals; none approved)

1. **Statistical power:** repeat runs (≥ 3 per configuration) or use a fixed-candidate offline
   evaluation before attributing gaps under 5 pp to a method.
2. **CG-faithful variant:** likelihood inputs, a separate "incorrect" prompt, the paper's
   running-average payoff, and accept/reject kept outside the game as an explicit rule.
3. **Jev Discriminator:** align its question with the chat criterion, or score target and
   attributes separately from relations.
4. **Evidence quality:** centred, marked or best-view candidate images (an imagery change).
5. **Archiving:** keep the analysis scripts and labels (currently in the session scratchpad) in
   the repo or a `saved_sessions/` analysis folder.

Each item that changes behaviour requires approval under [AGENTS.md](../AGENTS.md) and must not
be tuned on benchmark success rate.
