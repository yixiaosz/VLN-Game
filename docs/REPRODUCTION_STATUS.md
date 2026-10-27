# Reproduction status

Last updated: **2026-10-27**. This page describes the latest recorded outcomes,
not live process monitoring. Read [AGENTS.md](../AGENTS.md) for current editing
rules, [LOCAL_SETUP.md](LOCAL_SETUP.md) for operation and
[CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md) for approval/history.

## Current state and next decision

The CLIP-only language baseline and direct-API **DeepSeek V4.1 Flash VLN-Game**
each completed **316/316** episodes. DeepSeek Game finished with 101 successes,
SR 0.319620, SPL 0.120965, DTG 4.640878 m, final runtime 14h36m43s. The process
exited and released the GPU; its actual exit code was not collected by a waiting
parent. All completion events and final logs are archived and verified. This is
a DeepSeek-labelled reproduction milestone, not a GPT-4o paper-metric replication.
A full
GPT-4o-mini headless VLM attempt completed **188/316**, then stopped on an OpenRouter
key-level USD 50 total-spend limit during episode 189 (HTTP 403). Its final
completed episode was logged at 8h24m33s. The failure is archived and verified;
the limit was not raised and that run was not restarted. GPT-4o-mini's full
316-episode completion target remains unmet.

The user authorized a separate full headless **DeepSeek V4.1 Flash** run using the
configured direct DeepSeek API. It started at 01:39 AEDT on 2026-10-27 and completed
all episodes on the same day. Monitoring used manual one-hour sleeps; user
discussion explicitly interrupted the final scheduled sleep, followed by direct
checks and archival. No background monitoring script was used.
See [the manual check log](../saved_sessions/hourly-monitoring-20261027.md) and
[the sealed DeepSeek archive](../saved_sessions/20261027-deepseek-v4.1-flash-headless-316ep/archive-summary.json).

**Next authorized work:** Generator/Ranking implementation and both three-episode
DeepSeek headless smoke runs have passed (change 18). The user authorized a full
headless 316-episode DeepSeek Generator run next, with the same manual one-hour
sleep/check monitoring until it ends, followed by archival. See the corresponding
`saved_sessions/` launch record for live state; this page does not imply a running
session before that launch record exists.
Completion of an experiment does not establish that the user reviewed or signed
off on its results; do not infer that approval from these records.

**Earlier stopped-run cause: parsing-output structure, now corrected under change
16.** In episode 71, the model returned:

```json
{"target": "couch", "landmark": ["table", "flower vase", []]}
```

The existing syntax check accepted the nested list. The agent added it to the
detector classes; GroundingDINO's `". ".join(classes)` raised
`TypeError: sequence item 3: expected str instance, list found`. This happened at
step 0, before any candidate search/render in that episode. Episode 71 has null
metrics and is excluded from the completed-episode averages.

The user approved and implementation now enforces a string `target` and optional
list of string landmarks inside the existing parsing retry loop. Invalid shapes
regenerate within the unchanged three-attempt budget; values are not repaired,
flattened, dropped or coerced. All 72 offline tests passed, and replaying the exact
bad response followed by a real-provider retry recovered with unchanged settings.
A separate natural-provider replay of the same dataset episode completed
successfully; see the evidence below. These targeted checks do not complete the 100-episode run.
Do not restart that billed run or alter other parsing/evaluation behavior without
separate approval.

## Active limitations

- **GUI native crash:** the thread/callback fixes passed short checks but did not
  establish endurance. A later GUI run segfaulted in `FEngine::loop` during episode
  52, after 51 completed episodes and about 3h13m. Exact native cause is unresolved.
- **Headless stability:** the 188-episode mini attempt reached over eight hours
  without an observed native renderer crash; the subsequent DeepSeek Game run
  completed all 316 episodes over 14h36m43s without a matching kernel crash/GPU/OOM
  event. This is observed completion of those workloads, not a guarantee that
  future native Filament runs cannot fail. GUI crash cause remains unresolved.
- **GroundingDINO phrase mapping:** an existing dependency warning was observed
  during episode 158. Unmatched decoded phrases deterministically receive class
  0 (despite the message saying random), which can affect map labels. Actual
  unmatched phrases/counts are not recorded, so impact is unquantified. The
  detector behavior was not changed during measurement. Its helper source is
  preserved in the new GPT-4o-mini archive as an archive-time observation.
- **Pixel fidelity:** headless reference camera matrices matched the local GUI
  exactly in six render comparisons. Composition/JPEG encoding matched for the
  same render buffers; native cross-backend renders have small pixel differences.
- **Timeout/accounting limits:** API timeouts bound socket inactivity, not absolute
  elapsed time. SDK-internal retries/billing are not individually traced. Native
  synchronous offscreen rendering has no enforceable Python deadline or isolation.

## Implemented capabilities (not pending tasks)

- Candidate-specific atomic JPEG publication, first-view/identity/hash validation
  and archival of the exact API input buffers: change 11.
- Candidate-local Discriminator aggregation, explicitly approved by the user:
  change 12, label `candidate_local_responses_v1`.
- Structured traces, derived terminal reports and API attempt diagnostics:
  changes 9, 10 and 14. Required sample failure aborts without partial aggregation.
- Candidate-only headless VLM with `-v 0`: change 15. Interactive GUI mode remains
  available with `-v 1`. Change 17 adds headless/GUI selection as the first
  `demo.py` question for ObjectNav, CLIP and VLM modes.
- Generator (`vlm`, one sample) and Ranking (`vlm_rank`, five-sample plurality)
  are independently selectable alongside unchanged/default Game (`vlm_game`):
  change 18 fixes baseline ID selection and adds method-labelled tracing. Ties
  follow candidate order with reject last; invalid baseline IDs abort without
  changing Game's interpretation. All 79 offline tests and five before/after
  Game comparisons passed. Both DeepSeek baseline smokes completed 3/3 with
  actual exit code 0 and verified images/decisions, not benchmark evidence.

Detailed implementation and validation evidence lives in the change history,
not this active-issue list. Parsing-only shape validation/regeneration is approved
in change 16; other model-output interpretation changes remain separate decisions.

### Parsing correction verification (change 16)

The [targeted test archive](../saved_sessions/smoke_tests/20261026-parsing-structure-episode71/)
separates two checks: an explicitly injected replay of the archived malformed
response followed by a live retry (passed), and one natural-provider headless
navigation episode with the same instruction, dataset ID, HM3D scene and start
pose (passed: 193 steps, success 1.0, SPL 0.120460, DTG 0.111536 m). Both use the
original effective GPT-4o-mini request settings. The natural episode exited 0 in
2m10s including startup, with a first-attempt valid parsing response. It made no
candidate searches/vision calls and therefore does not add rendering/equilibrium
coverage. It is a standalone regression test; do not combine it with the old
70-episode benchmark aggregate.
The one-episode subset exists only in the test harness. It does not change
production episode selection or replay the historical process's RNG/API history.
The initial harness's first-scene comparison stopped before navigation; its failed
preflight and the completed parsing-recovery evidence are retained separately.
The [validation summary](../saved_sessions/smoke_tests/20261026-parsing-structure-episode71/validation.json)
records both successful checks and this harness-only preflight failure. No original
session evidence was modified, and no 100-episode run was restarted.

## Experiment milestones

SR/SPL/DTG values below are those recorded for completed episodes only. Rows are
different revisions/providers/workloads, not controlled evidence of improvement.
The paper used GPT-4o; GPT-4o-mini results must keep their actual model label.

| Milestone | Completed | SR | SPL | DTG (m) | Outcome / evidence |
|---|---:|---:|---:|---:|---|
| CLIP language baseline | 316/316 | 0.247 | 0.093 | 4.923 | Completed, 78 successes; [session record](../saved_sessions/20261025-clip-baseline-316of316/session.json) |
| Legacy GUI VLM (model not recorded) | 38/100 | 0.263 | 0.056 | 4.681 | Native renderer crash; historical uncorrected inputs/aggregation; [index](../saved_sessions/README.md) |
| Corrected-image/aggregation GPT-4o-mini GUI, before API fix | 20/100 | 0.400 | 0.077 | 5.287 | Stalled awaiting API sample, later terminated; [archive](../saved_sessions/20261025-gpt-4o-mini-vlm-game-20of100-terminated/session.json) |
| API-corrected GPT-4o-mini GUI | 51/100 | 0.254902 | 0.091772 | 3.850896 | Episode 52 native renderer crash; [summary](../saved_sessions/20261026-gpt-4o-mini-vlm-game-51of100-render-crashed/session.json) |
| GPT-4o-mini headless smoke | 3/3 | 0.333333 | 0.088991 | 3.414690 | Passed in 6m40s; not benchmark evidence; [audit](../saved_sessions/smoke_tests/20261026-headless-mini-smoke/validation.json) |
| GPT-4o-mini headless target attempt | 70/100 | 0.314286 | 0.108471 | 3.543361 | 22 successes; episode 71 parser error; [validated summary](../saved_sessions/20261026-gpt-4o-mini-vlm-game-headless-100ep/archive-summary.json) |
| GPT-4o-mini headless full-run attempt, parsing-corrected | 188/316 | 0.292553 | 0.104405 | 4.130340 | 55 successes; episode 189 HTTP 403 key limit; [verified archive](../saved_sessions/20261027-gpt-4o-mini-headless-188of316-key-limit/session.json) |
| DeepSeek V4.1 Flash Game, direct API/headless | 316/316 | 0.319620 | 0.120965 | 4.640878 | 101 successes; [sealed summary](../saved_sessions/20261027-deepseek-v4.1-flash-headless-316ep/archive-summary.json) |

Paper Table III's CLIP baseline reports SR 0.253, SPL 0.106, DTG 4.697. The local
full CLIP run is available for comparison; its completion supersedes old "baseline
pending" notes, without implying equality with the paper or user acceptance.

### Cancelled renderer-only test

The user explicitly replaced the proposed four-hour renderer-only endurance test
with the real 100-episode headless run. It stopped by SIGINT after 186 renders /
185.58 seconds. The generic script `error` status and exit -2 reflect cancellation,
not a renderer failure. No four-hour result exists. See
[the cancellation record](../saved_sessions/smoke_tests/20261026-headless-render-endurance/CANCELLED.md).
Do not reinstate this test as an active requirement without a new user request.

## Archived verification scope

The [headless session README](../saved_sessions/20261026-gpt-4o-mini-vlm-game-headless-100ep/README.md)
and [failure analysis](../saved_sessions/20261026-gpt-4o-mini-vlm-game-headless-100ep/FAILURE.md)
describe the complete preserved evidence. All 681 exact candidate input JPEGs
were decoded/hashed and matched their published files and trace metadata;
launch-source hashes and the 70-episode metric averages were verified. The
archive inventories 18,407 files. Canonical data was not repaired or rewritten.

This long-run archival audit did **not** replay all numerical policy checkpoints.
The separate three-episode headless smoke did pass image-hash and policy-replay
audits. The offline suite passed 68 tests at headless implementation time; that
historical observation is not a claim that every subsequent edit was retested.

`saved_sessions/` is local and ignored by Git. Links above resolve in this checkout;
a fresh clone will not contain these archives. Keep run-specific detail in those
archives and use the [saved-session index](../saved_sessions/README.md) to find it.

The [188/316 archive](../saved_sessions/20261027-gpt-4o-mini-headless-188of316-key-limit/README.md)
preserves all 189 episode outcomes (last error/incomplete), exact image/hash
evidence, original/run-only evaluation logs and launch-commit source/docs.
All 2,043 inventoried files and 47,843 similarity-map tar members were verified.
Similarity maps use a documented mtime window in shared legacy paths. The
already-running process's actual exit code and complete historical console
stdout cannot be recovered by the assistant; application stop logs and canonical
traces remain intact. No policy checkpoints were recalculated in this archival.

The [full DeepSeek Game archive](../saved_sessions/20261027-deepseek-v4.1-flash-headless-316ep/README.md)
retains all 316 closed episode traces/reports, full stdout/stderr, 2,818 exact
input images and 83,799 similarity maps. All 90,098 inventoried files were
verified. Ten whitespace-response API retries recovered; no terminal API-sample
failure was recorded. Image hashes match the published JPEGs and launch-source
hashes match the manifest. This archive did not replay all policy checkpoints.

Change-18 evidence is under
[baseline validation](../saved_sessions/smoke_tests/20261027-baseline-method-validation/),
[Generator smoke](../saved_sessions/smoke_tests/20261027-deepseek-generator-smoke/validation.json)
and [Ranking smoke](../saved_sessions/smoke_tests/20261027-deepseek-ranking-smoke/validation.json).
Each smoke's decisions were independently recomputed from the actual votes;
there were no Discriminator calls or equilibrium updates. The Game preservation
check compared against the sealed full-run source, including multi-candidate
checkpoint arrays. No historical archive was modified by baseline implementation.

## Maintenance

After a milestone or blocker change, replace the active summary/issue rather than
appending progress logs. Add a dated milestone with model, correction/backend
labels, completed/requested counts, outcome and evidence link. Distinguish
observed completion from interpretation, approval and unproven stability claims.
Update the root guide's brief status line only if the next action changes.
Follow the [documentation maintenance rules](../AGENTS.md#documentation-maintenance).
