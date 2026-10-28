# Agent guide

## Purpose and document map

Faithfully reproduce VLN-Game's zero-shot object/language-goal navigation on Habitat/HM3D
using the original GroundingDINO, MobileSAM, CLIP, object-centric mapping, frontier
exploration and game-theoretic VLM pipeline. Unrelated refactors/redesigns/tuning are outside scope.

Read this guide for every task; load the following documents when relevant:
| Document | Authority / when to read |
|---|---|
| [LOCAL_SETUP.md](docs/LOCAL_SETUP.md) | Environment, dependencies, data, API configuration and launch troubleshooting |
| [CHANGES_FROM_ORIGINAL.md](docs/CHANGES_FROM_ORIGINAL.md) | Approval/provenance and historical validation; read before modifying affected pipeline behavior |
| [REPRODUCTION_STATUS.md](docs/REPRODUCTION_STATUS.md) | Current blockers, milestone results and evidence; read before planning a fix or evaluation |
| [Local README](README.md) | User-facing operation and trace/report commands |
| [Saved-session index](saved_sessions/README.md) | Local experiment artifacts; ignored by Git and unavailable in a fresh clone |

## Editing and execution rules

1. **Fidelity first.** Do not alter algorithms, hyperparameters, evaluation, metrics,
   observation/action spaces or dataset semantics to improve results.
   Never tune thresholds, prompts, episode selection or configs for higher SPL/SR.
2. **Only fix blockers.** Make minimal install/import/data/runtime compatibility
   fixes or explicitly approved changes. No model-stack replacements or unrelated cleanup.
3. **Discuss behavioral changes first.** Obtain user approval before changing
   navigation, evaluation, metrics, model-output interpretation or GUI interaction.
   Approval of one correction is not permission for unrelated changes.
4. **Preserve protected surfaces.** Leave `requirements.txt`, `arguments.py`,
   `utils/vis_gui.py` and manual-query/pause behavior unchanged unless a documented
   blocker requires a change; behavioral changes still require approval.
5. **Preserve the original README.** Do not edit the text below the marker
   `(Below this line is the original README of the [VLN-game](https://github.com/ybgdgh/VLN-Game) code)`.
6. **No default long evaluation.** Use a 3–5 episode smoke run unless the user
   explicitly requests a larger run. Full benchmarks are 2,000 ObjectNav or 316
   language episodes. Do not restart a stopped billed run without authorization.
7. **Freeze during measurement.** During an explicitly agreed baseline/reference
   run, do not change logic. Completion does not imply user review/sign-off.
   Old baseline-first holds are historical; follow later approvals/current instructions.
8. **No sudo.** List necessary privileged commands for the user to execute.
9. **No unsolicited Git publication.** Commit, amend, push or create PRs only when requested.
10. **No secrets.** Never put keys, HM3D credentials, tokens or passwords in files,
    logs or argv, or print the private API environment's contents.
11. **Preserve user work/data.** Do not overwrite unfamiliar changes or delete datasets,
    tarballs or session evidence without explicit permission.
12. **Document persistent changes.** Record approval, scope, verification and evidence
    in the change history; follow the maintenance routing below.

## Reproduction and provenance invariants

- New supported VLM runs use request-scoped atomic candidate publication and
  exact-buffer image hashing/archival. Preserve run/episode/search/candidate and
  selected-first-view ownership; never substitute stale/shared `Vis-*.jpeg` files.
- Preserve `candidate_local_responses_v1`: each eligible candidate's Discriminator
  pool contains its own responses (approved 2026-10-25, superseding the GPT-4o reference hold).
- Required API-sample exhaustion uses `abort_test_no_partial_batch`: stop before
  partial aggregation; retain completed episodes and exclude incomplete episodes
  from metrics. Preserve historical failed-sample denominators in old evidence.
- Parsing-only shape validation/regeneration is approved (change 16): preserve
  string target/optional string-list landmarks within the existing retry budget.
  Unrecognized verdict handling, clamping and other interpretation changes remain separate decisions.
- Preserve old image/aggregation/backend labels, raw responses, nonfinite values
  and metrics. Historical defects do not establish which source produced the
  paper's metrics. Different providers/models and correction levels stay labelled.
- Offscreen rendering still uses Filament; short smoke checks do not establish
  multi-hour stability. Socket-inactivity timeouts are not absolute deadlines,
  and native offscreen calls have no enforceable Python timeout/crash isolation.

## Jev engineering decisions — approved/implemented 2026-10-28 (change 20)

- Add `typesafe/jev-1.14` as an isolated OpenRouter Decisions backend, not
  `typesafe/jev-router`. Preserve other VLMs' requests, live parsing, aggregation
  and equilibrium behavior; model switching must not leak Jev settings.
- Retain names/flags: **VLN-generator** (`vlm`), **VLN-ranking** (`vlm_rank`),
  **VLN-game** (`vlm_game`); identify Jev/version/aggregation explicitly in traces.
- **Jev-only frozen parsing:** DeepSeek generates an exact-original-instruction
  lookup to string `target` and string-list `landmark` using the existing parser
  prompt/settings/validation/retry budget. Dataset inspection found 316 unique
  instructions for 316 episodes. No key normalization or repeated sampling to pick a parse.
  Preserve raw outputs/provenance; resume completed entries, verify full coverage,
  then freeze/version/hash the table. Missing entries fail preflight; no live fallback.
  Structural validity is not proof of semantic accuracy. Other VLMs still parse live.
- Parsed phrases feed existing perception/target similarity only. Jev receives the unchanged full
  instruction, actual candidate IDs and exact archived JPEG composites, not extra
  parsed target/landmark hints. Build native `state`/`questions` deterministically;
  omit unsupported chat-generation parameters, including temperature.
- Generator: one Choice distribution. Ranking: equal-weight arithmetic mean of
  five complete Choice distributions. Select pooled argmax, candidate-order ties,
  reject (`-1`) last. Confidence is metadata, not a weight/threshold; no dedicated cost tracking.
- Game: mean of five Choices over **all offered candidates plus rejection**;
  mean of five candidate-local Nouls per real candidate, retaining `[d, 1-d]`.
  Multiple candidates: five all-image rejection Nouls; one candidate: complement
  its match probability. Do not normalize match probabilities across candidates.
- **Use arrays rather than the original equilibrium dictionaries.** With `K=N+1`
  options, `G` is `2×K` (`correct`/`incorrect` rows), `D` is `K×2` (verdict columns).
  Build `G=[g; 1-g]`, `D[i]=[d_i, 1-d_i]`; add bias `0.05`, row-normalize both.
  Initialize zero Q arrays. For `t=1..500`, accumulate
  `Q_G += D_previous.T/(2*t)` and `Q_D += G_previous.T/(2*t)` using previous
  **policy** matrices. Apply row-wise stable softmax to
  `(Q + lambda*log(initial_policy))/(lambda + 1/(eta*t))`, all lambdas/etas `0.1`.
  Select final Generator `correct`-row argmax. No column initialization,
  running-average payoff replacement, 5,000 iterations or sampled-support early exit.
- Preserve exact image/ID ownership, raw responses, pooled vectors and matrix
  axes/checkpoints. Require complete valid batches within existing bounded retries;
  never zero-fill missing options or aggregate partial batches. Verify array-solver
  parity, existing-model regressions and short smokes; no full benchmark is implied.

## Essential operation

Use single-process `main_vis.py` / `main_vln.py`, not vector variants. From the repository root:
```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vlngame
export GSA_PATH="$HOME/Grounded-Segment-Anything"
python main_vis.py --episode_count 5 -v 1
python main_vln.py --vln_mode clip --episode_count 5 -v 0
# Load private API config and select the effective VLN_* model/provider first.
source "$HOME/.config/vlngame/api.env"
python main_vln.py --vln_mode vlm_game --episode_count 5 -v 0
```

For VLM, `-v 0` is candidate-only headless (no DISPLAY); `-v 1` uses GUI.
`demo.py` asks headless/GUI first; VLM mode selects Generator/Ranking/Game and a registry model (change 21).
Direct CLI needs effective `VLN_*`, not just `VLN_MODELS`; see
[API configuration](docs/LOCAL_SETUP.md#private-api-configuration).
Existing chat methods: `vlm` = one sample, `vlm_rank` = five-sample plurality,
`vlm_game` = equilibrium (default). Baseline ties use candidate order, reject last.
Baseline ID failures share three request attempts (change 19); retries must not
add votes or change Game decoding. Jev's approved differences are specified above.
Smoke checks verify loading/completed metrics and either GUI controls or exact
headless inputs/no window. Offline code checks: `python -m unittest discover -s tests -v`.

Latest recorded state (2026-10-28): CLIP and DeepSeek Game, Generator and Ranking
completed 316/316 and are archived; mini stopped at 188/316 on a key limit. Sequential runs/manual monitoring are complete.
[status page](docs/REPRODUCTION_STATUS.md). Jev backend/table are implemented; check smoke verification there.

## Documentation maintenance

Keep this guide approximately **120–160 lines**: an operational policy/index, not a diary.

| What changed? | Update here |
|---|---|
| Active rules, approval boundaries, essential commands or document routing | `AGENTS.md` |
| Verified environment, pins, asset paths or detailed invocation | `docs/LOCAL_SETUP.md` |
| Approved persistent source/data/compatibility change | `docs/CHANGES_FROM_ORIGINAL.md` (append the next change ID) |
| Current blocker, completed milestone, cancelled/failed run or changed priority | `docs/REPRODUCTION_STATUS.md` |
| User-facing usage affected | Editable local section of `README.md` |
| Detailed logs, images, raw responses, diagnostics, checksums or full run analysis | The relevant `saved_sessions/` archive and its index/catalog |

1. **Read source/evidence first.** Date observations; distinguish implemented, approved-but-unimplemented,
   proposed, cancelled, failed and unknown. Proposals/old approvals/cancelled tests are not current instructions.
2. **One authoritative home.** Keep details in routed documents; refresh linked summaries when state changes.
3. **Stable history.** Preserve IDs/dates/approval scope/results/caveats; qualify
   superseded claims as historical and append corrections, never rewrite run behavior.
4. **Current status.** Replace resolved blockers/stale pending claims with outcomes
   and evidence; keep milestone history outside active blockers, no episode/PID diary here.
5. **Sealed archives stay unchanged.** Never update archived snapshots, manifests, images or checksums.
6. **Durable evidence.** Prefer `saved_sessions/` over `/tmp`; verify links and mark
   missing historical evidence explicitly; local archives need not exist in fresh clones.
7. **Docs-only verification.** Check links/anchors and `git diff --check`; confirm original README,
   source/data and archives unchanged. No model runs/repeated app tests or new source-change ID for docs-only work.
