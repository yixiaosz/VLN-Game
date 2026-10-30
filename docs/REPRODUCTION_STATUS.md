# Reproduction status

Last updated: **2026-10-31**. This page describes the latest recorded outcomes,
not live process monitoring. Read [AGENTS.md](../AGENTS.md) for current editing
rules, [LOCAL_SETUP.md](LOCAL_SETUP.md) for operation and
[CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md) for approval/history.

## Current state and next decision

**Archive folder names simplified by the user (2026-10-30, change 23).** Current
links/catalog use the shortened names. The external
[relocation map](../saved_sessions/session-renames.json) resolves names in dated
cleanup records and historical absolute paths. All eight existing seals and
37,624 inventory entries remain valid; no archive bytes or metrics changed and
no resealing was necessary.

**Approved cleanup completed (2026-10-30, change 22):** eight archives retained
(seven full benchmarks plus mini 188/316), all post-cleanup seals verified:
**37,624 inventory entries** and **37,573 protected pre-cleanup files** checked.
Retained measurements and protected raw bytes are unchanged. Standalone short/
failed/smoke directories, heatmap PNG trees and two equivalent compressed
heatmap archives/manifests are pruned; lookup generation moved into Jev Game.
`saved_sessions/` decreased from about 13 GiB to 2.1 GiB. Future final archives
follow the same no-heatmap retention policy; use the read-only verifier below.

**Full Jev VLN-generator completed (2026-10-30):** 316/316, 95 successes,
SR **0.300633**, SPL **0.081035**, mean DTG **4.376207 m**, logged runtime
**14h54m39s**. Actual exit **0** collected at 21:54:21 AEDT. All 2,965 exact
API input images/request hashes and 904 one-Choice argmax/tie selections verified.
90,827 similarity maps originally exported (pruned under change 22); 905 attempts, one recovered structural retry,
no terminal failure, zero Discriminator/equilibrium work. Pipeline/table hashes
match Ranking/Game, docs-only dirty state labelled; no logic change during run.
Ranking's 99,047-file archive was sealed first and Generator launched immediately
afterward. Generator's **original acquisition seal** covered 97,436 verified
files; current post-cleanup inventories are listed in the cleanup record below.
Manual 30-minute checks and user interruptions are recorded. Evidence:
[Generator session](../saved_sessions/20261030-jev-1.14-vln-generator-316ep/README.md).

All requested Jev full runs/manual monitoring are complete. No additional paid
run is implied or authorized by completion. Imagery/controller/rejection and
provider-rounding compatibility follow-ups remain separate decisions, not fixes
silently applied to these measurements. Run completion does not imply user sign-off.

**Full Jev VLN-ranking completed (2026-10-30):** 316/316, 87 successes,
SR **0.275316**, SPL **0.074048**, mean DTG **4.734162 m**, logged runtime
**14h51m43s**. Actual exit **0** collected at 06:44 AEDT. All 3,145 exact API
input images/request hashes and all 967 five-Choice means/selections verified.
92,073 similarity maps originally exported (now pruned); 4,883 attempts, 48 recovered structural retries,
no terminal failures, zero Discriminator calls/equilibrium updates. Pipeline
and frozen table hashes match Game exactly; launch dirty state was docs-only.
No imagery/rejection/controller/validation changes during measurement. Evidence:
[Ranking session](../saved_sessions/20261029-jev-1.14-vln-ranking-316ep/README.md).

At the user's request, manual 30-minute Ranking monitoring began at 23:44 AEDT
on Oct 29 and continued until exit. Ranking's original seal verified 99,047 files;
that is an acquisition count, not the revised post-cleanup inventory. The full Generator above
was launched immediately afterward; preserve the same pipeline/table.

**Full Jev VLN-game completed (2026-10-29):** 316/316 episodes, 68 successes;
SR **0.215190**, SPL **0.063983**, mean DTG **4.908385 m**, logged runtime
**18h28m34s**. Actual exit code **0** collected at 15:28:19 AEDT; GPU released.
Launched from clean committed `f60a8eb`, frozen table v1, original iterator/seeds
and locked Jev Game settings; no logic changes during measurement. All 4,623
exact input images/request hashes, all probability means/selections and 6,090
policy checkpoints were independently verified. 106,102 similarity maps were originally exported and are now pruned;
33,926 application attempts, 81 recovered structural retries, no terminal failure.
See [the verified Game archive](../saved_sessions/20261028-jev-1.14-vln-game-316ep/README.md).

Manual 30-minute monitoring, user interruptions and the user-requested pause
11:26–13:45 AEDT are recorded; no monitoring is claimed during that pause.
The separately authorized full Ranking run above followed Game archival. Keep
the committed settings/table. Updated images, forced selection or altered
rejection/controller policies remain proposals.

The user's 2026-10-29 executive review request is recorded in the running
session's [228-episode diagnostic report](../saved_sessions/20261028-jev-1.14-vln-game-316ep/diagnostics/EXECUTIVE_REVIEW_228.md).
On those matched instructions, Jev Game has 51 successes versus DeepSeek Game's
69; it rejects 90.3% of queries, and 74 all-rejection episodes with submitted
candidates hit 500 steps, consuming 6.83 hours. This is a provisional running
subset, not the full-run result. A diagnostic retry-counting error was corrected:
58 recovered structural retries (15 through episode 34), not zero; no terminal
failures through episode 228. Strict handling of rounded provider outputs is
flagged for a later compatibility decision; no pipeline change during measurement.

**Launcher update (2026-10-28, change 21):** `demo.py` now offers VLN-generator,
VLN-ranking and VLN-game for Jev and chat models, with Game as the Enter default.
Summary, CLI flag and printed log path follow the selection. All 21 mocked
launcher cases and 98 offline tests passed; no new navigation/API run was needed.

**Jev integration (2026-10-28, change 20): implemented and validated; all short smokes completed.**
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
accuracy or benchmark equivalence. No full Jev benchmark had been authorized at
that smoke milestone; the later explicit authorization above supersedes that hold. Evidence:
[retained lookup generation](../saved_sessions/20261028-jev-1.14-vln-game-316ep/configuration/parse-generation/).
The standalone backend smoke archive was intentionally retired by change 22;
the results above describe its historical validation, not current file retention.

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
See [the manual check log](../saved_sessions/20261027-deepseek-v4.1-flash-316ep/manual-monitoring.md) and
[the sealed DeepSeek archive](../saved_sessions/20261027-deepseek-v4.1-flash-316ep/archive-summary.json).

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
originally preserved separately (now retired under change 22); its metrics are not merged into the completed benchmark.

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

## Archive retention and cleanup

Approved **2026-10-30**, change 22: retain seven completed full runs and the
188/316 GPT-4o-mini attempt. Standalone short/failed/smoke captures are intentionally
removed; their historical results remain below, but their original files are no
longer claimed accessible. Raw DeepSeek parse generation is consolidated inside
Jev Game `configuration/parse-generation/`. All per-step heatmap PNG trees,
compressed equivalents and heatmap member manifests are removed, including from
future final archives. Exact candidate images, responses/traces, source/config
and full-run diagnostics/monitoring are preserved. Canonical run-only logs exclude
foreign launch blocks; duplicated root monitoring is consolidated into sessions.

Original similarity-map/inventory counts in dated validation describe acquisition,
not current retained files. Retired seal digests, preservation checks, intentional
removals and new post-cleanup seals are recorded in the local
[cleanup record](../saved_sessions/cleanup-record.json) and each session's
`cleanup-record.json`. Use the read-only verifier documented in
[local setup](LOCAL_SETUP.md#archive-retention-and-post-run-cleanup), not old
archival writers. No navigation rerun, raw-value repair or metric change is part
of cleanup; preserved source/documentation snapshots remain as measured.

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

The historical targeted test archive (intentionally retired under change 22)
separated two checks: an explicitly injected replay of the archived malformed
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
preflight and completed parsing-recovery evidence were originally retained separately.
Its validation summary recorded both successful checks and the harness-only
preflight failure. No original session evidence was modified at validation time,
and no 100-episode run was restarted. The standalone evidence was later pruned.

## Experiment milestones

SR/SPL/DTG values below are those recorded for completed episodes only. Rows are
different revisions/providers/workloads, not controlled evidence of improvement.
The paper used GPT-4o; GPT-4o-mini results must keep their actual model label.

| Milestone | Completed | SR | SPL | DTG (m) | Outcome / evidence |
|---|---:|---:|---:|---:|---|
| CLIP language baseline | 316/316 | 0.247 | 0.093 | 4.923 | Completed, 78 successes; [session record](../saved_sessions/20261025-clip-baseline-316ep/session.json) |
| Legacy GUI VLM (model not recorded) | 38/100 | 0.263 | 0.056 | 4.681 | Native renderer crash; historical uncorrected inputs/aggregation; [archive intentionally retired](#archive-retention-and-cleanup) |
| Corrected-image/aggregation GPT-4o-mini GUI, before API fix | 20/100 | 0.400 | 0.077 | 5.287 | Stalled awaiting API sample, later terminated; [archive intentionally retired](#archive-retention-and-cleanup) |
| API-corrected GPT-4o-mini GUI | 51/100 | 0.254902 | 0.091772 | 3.850896 | Episode 52 native renderer crash; [archive intentionally retired](#archive-retention-and-cleanup) |
| GPT-4o-mini headless smoke | 3/3 | 0.333333 | 0.088991 | 3.414690 | Passed in 6m40s; not benchmark evidence; [archive intentionally retired](#archive-retention-and-cleanup) |
| GPT-4o-mini headless target attempt | 70/100 | 0.314286 | 0.108471 | 3.543361 | 22 successes; episode 71 parser error; [archive intentionally retired](#archive-retention-and-cleanup) |
| GPT-4o-mini headless full-run attempt, parsing-corrected | 188/316 | 0.292553 | 0.104405 | 4.130340 | 55 successes; episode 189 HTTP 403 key limit; [verified archive](../saved_sessions/20261027-gpt-4o-mini-188ep/session.json) |
| DeepSeek V4.1 Flash Game, direct API/headless | 316/316 | 0.319620 | 0.120965 | 4.640878 | 101 successes; [sealed summary](../saved_sessions/20261027-deepseek-v4.1-flash-316ep/archive-summary.json) |
| DeepSeek V4.1 Flash Generator, direct API/headless | 5/316 | 0.600 | 0.206 | 2.093 | 3 successes; episode 6 unoffered ID error; [archive intentionally retired](#archive-retention-and-cleanup) |
| DeepSeek V4.1 Flash Generator, bounded-ID retry/headless | 316/316 | 0.322785 | 0.116120 | 4.903238 | 102 successes; [sealed summary](../saved_sessions/20261027-deepseek-v4.1-flash-generator-316ep/archive-summary.json) |
| DeepSeek V4.1 Flash Ranking, bounded-ID retry/headless | 316/316 | 0.322785 | 0.122272 | 4.890288 | 102 successes; [sealed summary](../saved_sessions/20261028-deepseek-v4.1-flash-ranking-316ep/archive-summary.json) |
| Jev 1.14 Game, native probability means/frozen parsing/headless | 316/316 | 0.215190 | 0.063983 | 4.908385 | 68 successes, actual exit 0; [verified summary](../saved_sessions/20261028-jev-1.14-vln-game-316ep/archive-summary.json) |
| Jev 1.14 Ranking, five-Choice mean/frozen parsing/headless | 316/316 | 0.275316 | 0.074048 | 4.734162 | 87 successes, actual exit 0; [verified summary](../saved_sessions/20261029-jev-1.14-vln-ranking-316ep/archive-summary.json) |
| Jev 1.14 Generator, one Choice/frozen parsing/headless | 316/316 | 0.300633 | 0.081035 | 4.376207 | 95 successes, actual exit 0; [verified summary](../saved_sessions/20261030-jev-1.14-vln-generator-316ep/archive-summary.json) |

Paper Table III's CLIP baseline reports SR 0.253, SPL 0.106, DTG 4.697. The local
full CLIP run is available for comparison; its completion supersedes old "baseline
pending" notes, without implying equality with the paper or user acceptance.

### Cancelled renderer-only test

The user explicitly replaced the proposed four-hour renderer-only endurance test
with the real 100-episode headless run. It stopped by SIGINT after 186 renders /
185.58 seconds. The generic script `error` status and exit -2 reflect cancellation,
not a renderer failure. No four-hour result exists. See
the cancellation result above; the standalone cancellation record was intentionally retired by change 22.
Do not reinstate this test as an active requirement without a new user request.

## Archived verification scope

The earlier 70-episode headless archive's README/failure analysis (now retired)
described its original evidence. All 681 exact candidate input JPEGs
were decoded/hashed and matched their published files and trace metadata;
launch-source hashes and the 70-episode metric averages were verified. The
original archive inventoried 18,407 files. Data was not repaired or rewritten;
the whole short/failed archive was later intentionally pruned under change 22.

This long-run archival audit did **not** replay all numerical policy checkpoints.
The separate three-episode headless smoke did pass image-hash and policy-replay
audits. The offline suite passed 68 tests at headless implementation time; that
historical observation is not a claim that every subsequent edit was retested.

`saved_sessions/` is local and ignored by Git. Links above resolve in this checkout;
a fresh clone will not contain these archives. Keep run-specific detail in those
archives and use the [saved-session index](../saved_sessions/README.md) to find it.

The [188/316 archive](../saved_sessions/20261027-gpt-4o-mini-188ep/README.md)
preserves all 189 episode outcomes (last error/incomplete), exact image/hash
evidence, the canonical run-only evaluation log and launch-commit source/docs.
Originally, 2,043 files and 47,843 similarity-map tar members were verified;
the mixed appended log and compressed heatmaps/manifests were intentionally pruned.
Original map attribution used a documented mtime window in shared legacy paths. The
already-running process's actual exit code and complete historical console
stdout cannot be recovered by the assistant; application stop logs and canonical
traces remain intact. No policy checkpoints were recalculated in this archival.

The [full DeepSeek Game archive](../saved_sessions/20261027-deepseek-v4.1-flash-316ep/README.md)
retains all 316 closed episode traces/reports, full stdout/stderr, 2,818 exact
input images. Original acquisition included 83,799 now-pruned similarity maps
and 90,098 verified inventory files. Ten whitespace-response retries recovered; no terminal API-sample
failure was recorded. Image hashes match the published JPEGs and launch-source
hashes match the manifest. This archive did not replay all policy checkpoints.

Change-18 evidence originally included baseline validation and Generator/Ranking
smoke archives, all intentionally retired under change 22.
Each smoke's decisions were independently recomputed from the actual votes;
there were no Discriminator calls or equilibrium updates. The Game preservation
check compared against the sealed full-run source, including multi-candidate
checkpoint arrays. No historical archive was modified by baseline implementation.

The change-19 static regression (standalone archive now intentionally retired)
recorded exact ordered-image hashes, unchanged request settings, injected/live
attempt labels and the accepted ID/vote. It passed without navigation; Game
comparisons and the 84-test offline suite passed. No claim of resumed or completed
316-episode Generator metrics follows from that static check.

The [completed Generator archive](../saved_sessions/20261027-deepseek-v4.1-flash-generator-316ep/README.md)
contains full stdout, 316 closed traces/reports and 1,106 exact API inputs. Its
64,569 similarity maps were intentionally pruned. All 537 selections were recomputed against offered IDs and the
single accepted response. There were 854 application attempts (316 parsing +
537 Generator + one parsing retry), no terminal sample failures, no Discriminator
calls and no policy updates. The original 67,444 files passed its acquisition inventory.
Later approved cleanup prunes only documented artifacts; measurements remain unchanged. The detached exit code remains
unobserved; completion records and kernel evidence are preserved. Raw console
contains 129 inherited GroundingDINO fallback-warning occurrences; this is not a
count of unique objects or a quantified explanation of navigation failures.

The [completed Ranking archive](../saved_sessions/20261028-deepseek-v4.1-flash-ranking-316ep/README.md)
contains full stdout, 316 closed traces/reports and 1,098 exact API inputs. Its
64,923 similarity maps were intentionally pruned. All 526 selections, including 17 tied searches, were recomputed
from five accepted responses and candidate-order/reject-last ties. There were
2,957 application attempts (316 parsing + 2,630 Generator + 11 retries), no
terminal sample failures, Discriminator calls or equilibrium updates. All 67,782
files passed the original acquisition SHA-256 inventory. One invalid-ID and ten empty-content
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
