# Reproduction status

Last updated: **2026-10-28**. This page describes the latest recorded outcomes,
not live process monitoring. Read [AGENTS.md](../AGENTS.md) for current editing
rules, [LOCAL_SETUP.md](LOCAL_SETUP.md) for operation and
[CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md) for approval/history.

## Current state and next decision

**Launcher update (2026-10-28, change 21):** `demo.py` now offers VLN-generator,
VLN-ranking and VLN-game for Jev and chat models, with Game as the Enter default.
Summary, CLI flag and printed log path follow the selection. All 21 mocked
launcher cases and 98 offline tests passed; no new navigation/API run was needed.

**Jev integration (2026-10-28, change 20): implemented locally; all short smokes completed.**
The isolated `typesafe/jev-1.14` Decisions backend supports the same Generator,
Ranking and Game names/flags, with native equal-weight probability means for the
five-query methods. Other models retain live parsing and their existing methods.
The Jev-only DeepSeek table is frozen and independently verified: 316 exact
instructions, 316 accepted application attempts, no retries; SHA-256
`d4abcaaf62f8cdf107e9490a71b60af80446607839454f9d40b7b4a6ac83812a`.
Structural validity does not establish semantic accuracy. All 98 offline tests
passed; array/dictionary equation parity and five sealed-reference chat Game
comparisons passed. Each authorized three-episode Jev headless smoke completed
3/3 with actual exit code 0: Generator, Ranking and Game each had **0/3 navigation
successes**, SPL 0, mean DTG 7.748133 m. Each made nine searches with 13 exact input
images. Application attempts: Generator 9, Ranking 45, Game 130; no retries or
terminal failures. Independent audit verified all 39 archived image buffers and
their ownership/request hashes, every probability mean/selection, and all 45
Game policy checkpoints against the original dictionary equations. Game completed
500 updates per search. Functional completion is not evidence of navigation
accuracy or benchmark equivalence; no full Jev benchmark is authorized. Evidence:
[backend validation](../saved_sessions/smoke_tests/20261028-jev-backend-validation/)
and [lookup generation](../saved_sessions/smoke_tests/20261028-jev-lookup-generation/).

The CLIP-only language baseline and direct-API **DeepSeek V4.1 Flash VLN-Game**
each completed **316/316** episodes. DeepSeek Game finished with 101 successes,
SR 0.319620, SPL 0.120965, DTG 4.640878 m, final runtime 14h36m43s. The process
exited and released the GPU; its actual exit code was not collected by a waiting
parent. All completion events and final logs are archived and verified. This is
a DeepSeek-labelled reproduction milestone, not a GPT-4o paper-metric replication.
A full GPT-4o-mini headless VLM attempt completed **188/316**, then stopped on an OpenRouter
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

Generator/Ranking implementation and both three-episode DeepSeek headless smoke
runs passed (change 18) and were pushed as `1eb0267`. The authorized full Generator
run launched at 16:59 AEDT, then stopped at 17:08:07 after 5/316 completed episodes.
Episode 6/search 4 offered IDs `[0,3,18,19,21]`; the VLM returned `{"object_id":"4"}`
with HTTP 200. The baseline-only membership guard aborted before aggregation.
At that revision the guard was outside the request retry loop, so no semantic-ID regeneration
occurred. This is the documented fail-fast policy triggering, not a renderer
segfault or recurrence of the old ID/index selection bug. Episode 6 is incomplete
and excluded. Source/prompt/settings were unchanged, and the run was not restarted.
The first hourly sleep was interrupted by the user for the 17:21 failure check.
The user approved change 19: baseline-only invalid-ID validation now occurs inside
the existing shared three-attempt request budget. A static replay of the exact
failed request rejected the injected archived ID 4, then a live identical-payload
retry returned offered ID 18 as one accepted logical sample. All 84 tests and
before/after Game comparisons passed. The ID is not guessed as an ordinal or
converted into rejection. After commit/push, the explicitly authorized restart
used the same full headless DeepSeek Generator settings and normal beginning;
manual checks used 30-minute sleeps. The restart completed all 316 episodes on
Oct 28: 102 successes, SR 0.322785, SPL 0.116120, DTG 4.903238 m; runtime
10h43m12s. One parsing_structure retry recovered; no candidate_id retry occurred
in this natural run (that recovery was verified in the static test). All baseline
decisions, image/source hashes and 67,444 archive files were verified, with zero
Discriminator calls or equilibrium updates. The prior five-episode failure is
preserved separately; its metrics are not merged into this completed benchmark.

The separately authorized full headless DeepSeek Ranking run launched after
Generator archival verified and completed all 316 episodes on Oct 28. Final:
102 successes, SR 0.322785, SPL 0.122272, DTG 4.890288 m; runtime 10h36m10s.
Manual 30-minute sleep/check monitoring confirmed exit/GPU release, then sealed
the archive. All 67,782 files, source/image hashes, five-vote counts and all 526
plurality/tie decisions were verified. One invalid-ID and ten empty-content
retries recovered without adding votes. No terminal API sample failure or
matching kernel crash/GPU/OOM record was found. No Discriminator or equilibrium
work was performed. Generator/Ranking pipeline hashes match exactly; Ranking's
launch Git dirty state consists only of the documented status-document updates.
All requested sequential full runs and monitoring are complete; no additional
paid run is authorized by this outcome record.
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
  event. Generator and Ranking subsequently each completed 316/316 over about
  10h40m without a matching native crash record. These are observed workloads, not a guarantee that
  future native Filament runs cannot fail. GUI crash cause remains unresolved.
- **GroundingDINO phrase mapping:** an existing dependency warning was observed
  during episode 158. Unmatched decoded phrases deterministically receive class
  0 (despite the message saying random), which can affect map labels. Exact
  unmatched phrases/unique-object counts are not recorded; console warning-line
  counts are available for the captured full baseline runs, but impact is unquantified. The
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
  follow candidate order with reject last; invalid baseline IDs retry within the
  shared budget and abort on exhaustion (change 19), without changing Game decoding.
  All 79 change-18 offline tests and five before/after
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
| DeepSeek V4.1 Flash Generator, direct API/headless | 5/316 | 0.600 | 0.206 | 2.093 | 3 successes; episode 6 unoffered ID error; [archive](../saved_sessions/20261027-deepseek-v4.1-flash-generator-headless-316ep/README.md) |
| DeepSeek V4.1 Flash Generator, bounded-ID retry/headless | 316/316 | 0.322785 | 0.116120 | 4.903238 | 102 successes; [sealed summary](../saved_sessions/20261027-deepseek-v4.1-flash-generator-headless-316ep-idretry/archive-summary.json) |
| DeepSeek V4.1 Flash Ranking, bounded-ID retry/headless | 316/316 | 0.322785 | 0.122272 | 4.890288 | 102 successes; [sealed summary](../saved_sessions/20261028-deepseek-v4.1-flash-ranking-headless-316ep/archive-summary.json) |

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

The [change-19 static regression](../saved_sessions/smoke_tests/20261027-baseline-id-retry-static/README.md)
records exact ordered-image hashes, unchanged request settings, injected/live
attempt labels and the accepted ID/vote. It passed without navigation; Game
comparisons and the 84-test offline suite passed. No claim of resumed or completed
316-episode Generator metrics follows from that static check.

The [completed Generator archive](../saved_sessions/20261027-deepseek-v4.1-flash-generator-headless-316ep-idretry/README.md)
contains full stdout, 316 closed traces/reports, 1,106 exact API inputs and 64,569
similarity maps. All 537 selections were recomputed against offered IDs and the
single accepted response. There were 854 application attempts (316 parsing +
537 Generator + one parsing retry), no terminal sample failures, no Discriminator
calls and no policy updates. All 67,444 files passed the final SHA-256 inventory.
No original output was deleted or repaired. The detached process exit code remains
unobserved; completion records and kernel evidence are preserved. Raw console
contains 129 inherited GroundingDINO fallback-warning occurrences; this is not a
count of unique objects or a quantified explanation of navigation failures.

The [completed Ranking archive](../saved_sessions/20261028-deepseek-v4.1-flash-ranking-headless-316ep/README.md)
contains full stdout, 316 closed traces/reports, 1,098 exact API inputs and 64,923
similarity maps. All 526 selections, including 17 tied searches, were recomputed
from five accepted responses and candidate-order/reject-last ties. There were
2,957 application attempts (316 parsing + 2,630 Generator + 11 retries), no
terminal sample failures, Discriminator calls or equilibrium updates. All 67,782
files passed the final SHA-256 inventory. One invalid-ID and ten empty-content
retries recovered; each aggregate denominator remained five. Pipeline hashes
match Generator exactly. Status-only Git dirty state and unobserved detached
exit code remain explicitly labelled. Raw stdout contains 179 inherited detector
fallback-warning occurrences, not unique-object or causal failure counts.

Generator and Ranking each succeeded on 102 episodes; Game succeeded on 101.
Ranking's SPL point estimate is slightly higher than Game's, and Game has lower
mean DTG. These single-run point estimates do not establish method superiority
or paper-metric equivalence. Preserve model/provider/correction labels and raw
per-episode results for any later paired/statistical analysis.

## Maintenance

After a milestone or blocker change, replace the active summary/issue rather than
appending progress logs. Add a dated milestone with model, correction/backend
labels, completed/requested counts, outcome and evidence link. Distinguish
observed completion from interpretation, approval and unproven stability claims.
Update the root guide's brief status line only if the next action changes.
Follow the [documentation maintenance rules](../AGENTS.md#documentation-maintenance).
