# Agent guide

## Purpose and document map

Faithfully reproduce VLN-Game's zero-shot object/language-goal navigation on
Habitat/HM3D using the original GroundingDINO, MobileSAM, CLIP, object-centric
mapping, frontier exploration and game-theoretic VLM pipeline. Unrelated
refactors, redesigns and benchmark tuning are outside scope.

Read this guide for every task; load the following documents when relevant:

| Document | Authority / when to read |
|---|---|
| [LOCAL_SETUP.md](docs/LOCAL_SETUP.md) | Environment, dependencies, data, API configuration and launch troubleshooting |
| [CHANGES_FROM_ORIGINAL.md](docs/CHANGES_FROM_ORIGINAL.md) | Approval/provenance and historical validation; read before modifying affected pipeline behavior |
| [REPRODUCTION_STATUS.md](docs/REPRODUCTION_STATUS.md) | Current blockers, milestone results and evidence; read before planning a fix or evaluation |
| [Local README](README.md) | User-facing operation and trace/report commands |
| [Saved-session index](saved_sessions/README.md) | Local experiment artifacts; ignored by Git and unavailable in a fresh clone |

## Editing and execution rules

1. **Fidelity first.** Do not alter algorithms, hyperparameters, evaluation,
   metrics, observation/action spaces or dataset semantics to improve results.
   Never tune thresholds, prompts, episode selection or configs for higher SPL/SR.
2. **Only fix blockers.** Make minimal install/import/data/runtime compatibility
   fixes or explicitly approved corrections. Do not replace the model stack or
   clean up unrelated code, names, formatting or dead paths.
3. **Discuss behavioral changes first.** Obtain user approval before changing
   navigation, evaluation, metrics, model-output interpretation or GUI interaction.
   Approval of one correction is not permission for unrelated changes.
4. **Preserve protected surfaces.** Leave `requirements.txt`, `arguments.py`,
   `utils/vis_gui.py` and manual-query/pause behavior unchanged unless a documented
   blocker requires a change; behavioral changes still require approval. Existing
   approved corrections are recorded in the change history.
5. **Preserve the original README.** Do not edit the text below the marker
   `(Below this line is the original README of the [VLN-game](https://github.com/ybgdgh/VLN-Game) code)`.
6. **No default long evaluation.** Use a 3–5 episode smoke run unless the user
   explicitly requests a larger run. Full benchmarks are 2,000 ObjectNav or 316
   language episodes. Do not restart a stopped billed run without authorization.
7. **Freeze during measurement.** During an explicitly agreed baseline/reference
   run, do not change program logic. Run completion does not imply user review or
   sign-off. Old baseline-first holds are historical, not new active instructions;
   follow the documented later approvals and the user's current task.
8. **No sudo.** List necessary privileged commands for the user to execute.
9. **No unsolicited Git publication.** Commit, amend, push or create PRs only
   when explicitly requested.
10. **No secrets.** Never put keys, HM3D credentials, tokens or passwords in files,
    logs or argv. Do not print the private API environment's contents.
11. **Preserve user work and data.** Do not overwrite unfamiliar changes or delete
    datasets, tarballs or session evidence without explicit permission.
12. **Document approved persistent changes.** Record source/data compatibility
    changes in the change history, with approval, scope, verification and evidence.
    Follow the maintenance instructions below rather than appending logs here.

## Reproduction and provenance invariants

- New supported VLM runs use request-scoped atomic candidate publication and
  exact-buffer image hashing/archival. Preserve run/episode/search/candidate and
  selected-first-view ownership; never substitute stale/shared `Vis-*.jpeg` files.
- Preserve `candidate_local_responses_v1`: each eligible candidate's Discriminator
  pool contains its own responses. The user approved this algorithmic correction
  on 2026-10-25, superseding the earlier hold for a completed GPT-4o reference run.
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

## Essential operation

Use single-process `main_vis.py` / `main_vln.py`; do not use the vector variants.
From the repository root:

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

For VLM, `-v 0` is candidate-only headless rendering, with no DISPLAY needed;
`-v 1` uses the GUI. `demo.py` asks headless/GUI first and selects a registry
model for VLM mode. Direct CLI runs require effective `VLN_*` settings, not just `VLN_MODELS`;
see [API configuration](docs/LOCAL_SETUP.md#private-api-configuration).
VLM methods remain independently selectable: `vlm` = one Generator sample,
`vlm_rank` = five-sample plurality, `vlm_game` = existing equilibrium (default).
Baseline ties follow candidate order, with reject last; preserve Game behavior.

Check environment/model/data loading and completed episode metrics in smoke runs.
For GUI checks, verify the window and controls; for headless checks, verify the
no-window path and exact candidate inputs. Do not require a GUI for headless success.
Run relevant offline checks after code changes:

```bash
python -m unittest discover -s tests -v
```

Latest recorded state (2026-10-27): CLIP and DeepSeek Game completed 316/316;
headless mini stopped at 188/316 on an OpenRouter key limit. Generator/Ranking
are validated (change 18); a full DeepSeek Generator run is authorized next. See the
[status page](docs/REPRODUCTION_STATUS.md) before choosing next work.

## Documentation maintenance

Keep this guide approximately **120–160 lines**. It is an operational policy/index,
not a changelog or experiment diary. Use these destinations for future updates:

| What changed? | Update here |
|---|---|
| Active rules, approval boundaries, essential commands or document routing | `AGENTS.md` |
| Verified environment, pins, asset paths or detailed invocation | `docs/LOCAL_SETUP.md` |
| Approved persistent source/data/compatibility change | `docs/CHANGES_FROM_ORIGINAL.md` (append the next change ID) |
| Current blocker, completed milestone, cancelled/failed run or changed priority | `docs/REPRODUCTION_STATUS.md` |
| User-facing usage affected | Editable local section of `README.md` |
| Detailed logs, images, raw responses, diagnostics, checksums or full run analysis | The relevant `saved_sessions/` archive and its index/catalog |

When editing any of these documents:

1. **Read the affected source/evidence first.** Date observations and distinguish
   implemented, approved-but-unimplemented, proposed, cancelled, failed and unknown.
   Do not turn a proposal, old approval or cancelled test into a current instruction.
2. **Use one authoritative home.** Keep detailed setup, change rationale and run
   metrics in their designated documents. Link from others; repeat only a short
   operational summary. Update that summary when the underlying state changes.
3. **Keep history stable.** Preserve change IDs, dates, approval scope, original
   results and caveats. Qualify superseded statements as historical; append a
   correction rather than retroactively rewriting a run's behavior or approval.
4. **Keep status current.** Replace resolved active items and stale "pending"
   claims with outcomes/evidence links. Retain milestone history outside the active
   blocker list. Do not append per-episode progress or old PIDs to this guide.
5. **Preserve archives.** New status updates do not authorize edits to sealed
   session snapshots, manifests, images or checksum inventories. Never change an
   archived AGENTS/README snapshot to match today's documentation.
6. **Use durable evidence.** Prefer existing `saved_sessions/` records over `/tmp`
   links. Verify referenced files exist; mark missing historical evidence explicitly
   and do not invent replacements. Local archive links need not work in a fresh clone.
7. **Verify documentation-only scope.** Check local links and section references,
   run `git diff --check`, and confirm the original README section, source/data and
   archived evidence are unchanged. Documentation-only edits do not need model runs
   or a repeated application test suite. Do not create a new code-change ID merely
   for a documentation move.
