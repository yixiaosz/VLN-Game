# Changes from the original paper repository

This is the historical implementation/approval record, moved from AGENTS.md on
2026-10-26. Entries describe behavior and validation **at that revision**; later
entries may supersede earlier limitations. An old "pending" statement is not a
new instruction or a current task. Active rules live in [AGENTS.md](../AGENTS.md),
setup in [LOCAL_SETUP.md](LOCAL_SETUP.md), and current blockers/results in
[REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md).
Code and artifact paths in historical entries are repository-relative unless
explicitly absolute; relocating this document did not relocate those artifacts.

## Reading and maintaining this record

- Preserve all numbered change IDs and their approval chronology. Append new
  approved persistent changes with the next ID; do not renumber old entries.
- State approval/date, affected files, reason, behavioral scope, validation and
  evidence location. Distinguish algorithmic corrections from compatibility or
  observability changes. Include limitations and any superseded entry.
- Test counts and smoke metrics below are historical evidence, not the current
  test-suite size or completed benchmark claims. Old source defects do not prove
  which revision produced the paper's reported metrics.
- Local `saved_sessions/` archives are ignored by Git; links to them work only in
  this checkout. Do not edit their documentation snapshots or canonical evidence.
- Historical `/tmp/opencode/candidate-image-vlm-smoke/` and
  `/tmp/opencode/discriminator-corrected-mini-smoke/` directories were absent at
  reorganization. Their old locations and recorded observations are retained,
  but do not claim those artifacts are currently accessible or independently
  reverified. Later persistent archives do not replace evidence for earlier runs.
- Follow the [documentation maintenance rules](../AGENTS.md#documentation-maintenance).

## Discriminator response accumulation — approved algorithmic correction

**Status (2026-10-25):** implemented after explicit user approval superseded the
earlier hold for a completed 100-episode GPT-4o test. This is an algorithmic
correctness correction, not merely logging or an environment compatibility fix.
New manifests identify it as `candidate_local_responses_v1`; old uncorrected
archives remain unchanged. Validation is recorded in change entry 12.

**Location and origin:** `Equilibrium_Ranking._get_discriminative_probabilities()`
in `utils/equ_ranking.py`. The same accumulation defect is present in the
original paper-code commit `58ca7f4`; it predates the local change from JSON
discriminator responses to plain-text `correct` / `incorrect`. This source
history does not establish which revision produced the paper's reported metrics.

**1. What was the bug?**

- `responses = []` was initialized once, outside the per-candidate loop.
- Each candidate's five returned responses were appended to the same list.
- The scoring loop counted `correct` responses across that entire list,
  including responses about earlier candidates, but still divides by
  `num_samples` (normally five).
- Consequently, candidate A uses A's responses; candidate B uses A+B;
  candidate C uses A+B+C. This contamination is confined to one search
  invocation, not carried between searches or episodes.
- Legacy logging also re-logged the accumulated responses for every candidate,
  so those old response-line counts cannot reliably measure API requests or
  token use. The multi-candidate reject-all query's list was separately reset.

Simple example: A receives 4/5 `correct`, B receives 2/5. The buggy calculation
scored A as 0.8 and B as `(4 + 2) / 5 = 1.2`. B's own frequency is 0.4.

**2. Why must this be fixed before reproduction runs?**

- An empirical frequency can exceed 1.0. With the existing 0.05 bias, B's
  `[correct, incorrect]` values in the example became
  `[1.25, -0.15]`; normalization yields approximately `[1.13636, -0.13636]`.
  Subsequent `np.log()` of a negative policy entry can produce NaNs and
  invalidate later policy updates. Contamination also distorts valid-looking
  values within [0, 1], so range checks alone cannot identify every affected
  search. The intended bias producing an unnormalized 1.05 is not itself a bug.
- The defect applied when at least two non-reject candidates were individually
  discriminated. It did not contaminate the single-candidate branch, the
  Generator-only early rejection, or CLIP-only navigation. The number of
  input images is not the same as the number of discriminated candidates.
- Even finite, in-range contaminated values can change the equilibrium choice:
  the scores depend on earlier candidates instead of each candidate's evidence.
  Navigation trajectories and Success/SPL/DTG can consequently change. Leaving
  this in place would not reliably evaluate the intended equilibrium algorithm.
- Correction does not guarantee better metrics or equality with the paper's
  results. The inherited source bug does not prove the authors used that revision
  for their reported runs. This issue is separate from the native Open3D crash.

**3. What is the fix?**

Reset both `responses` and `response_ids` inside the eligible-candidate branch,
immediately before submitting that candidate's five queries. Resetting the IDs
along with the responses ensures trace sample ownership is correct. The scoring
pool now contains only that candidate's returned samples; the example becomes
A=4/5=0.8 and B=2/5=0.4. Averaging the accumulated list or clamping the resulting
values would not correct the cross-candidate contamination.

The five-sample protocol, prompts, bias, equilibrium equations, candidate mapping,
single-candidate complement, multi-candidate reject-all query, fixed-five
Discriminator denominator, and invalid-response handling are otherwise preserved.
The later transport fix aborts on a required sample error before any partial-batch
frequency is produced; it does not change these successful-batch formulas.
No clamping is added. Tests cover two/three candidates, candidate-order independence
of mapped frequencies, single-candidate and all-reject branches, failed samples,
and finite/nonnegative/normalized initial and checkpoint policies. Reports retain
legacy contaminated/NaN fixtures rather than rewriting historical evidence.

## Historical API failure investigation

Before change 14, the corrected GPT-4o-mini run completed 20/100 episodes, then
stalled during episode 21/search 8 awaiting Discriminator sample `d-36-5`. Two
earlier `JSONDecodeError` events (episode 7/search 5 Generator; episode 21/search 1
Discriminator) came from decoding the outer HTTP response, not the model's
`object_id` text. HTTP status/body diagnostics were not captured, so the exact
upstream cause cannot be recovered. At that revision, `requests.post` had no
timeout/status check, caught the wrong exception family, and malformed Generator
content did not consume the retry budget.

The lingering process was stopped before verified archival. See the
[20/100 session](../saved_sessions/20261025-gpt-4o-mini-vlm-game-20of100-terminated/session.json)
and its `API-error-investigation.txt`. No API behavior changed during that
archival/investigation task; the later reliability correction is change 14.

## Changes from the original paper repo

The commit `58ca7f4` is a clean fork from the original paper's code repo.

1. **`main_vis.py`** — removed the forced `action = 0` for the first 5
   episodes.
   - Reason: the original debug bypass prevents the agent from running the
     first 5 episodes. It blocks a 3-5 episode GUI smoke test and would
     corrupt a full ObjectNav evaluation.

2. **`main_vln.py`** — removed the forced `action = 0` for the first 253
   episodes.
   - Reason: same debug bypass; it prevents language navigation from
     starting for most of the 316-episode validation set.

3. **`utils/detection_segmentation.py`** — removed the unused
   RAM/Tag2Text import block.
   - Reason: this implementation leaves the RAM/Tag2Text initialization and
     inference blocks commented out; the paper covers DINO and Mobile-SAM.
     The import required an unavailable package.
   - The dead paths/constants and commented code were left untouched to keep
     the diff minimal.

4. **`utils/chat_utils.py`** — generic OpenAI-compatible API configuration.
   - Reads `VLN_API_KEY`, `VLN_API_BASE`, `VLN_MODEL`,
     `VLN_MODEL_OPTIONS`, and `VLN_MODEL_SUPPORTS_TEMPERATURE` from the
     environment, with the original OpenAI defaults preserved.
   - Supports OpenRouter GPT-4o, GPT-4o mini, and GPT-6 Luna, plus direct
     DeepSeek `deepseek-flash`.
   - GPT-6 Luna uses `reasoning.effort=none` and omits temperature. DeepSeek
     uses `thinking.type=disabled`.
   - The parsing request has a 512-token output cap. OpenRouter otherwise
     reserves a large default output budget and rejects low-credit keys.
   - Reason: select a language model without putting keys in `argv` or the
     repository. The paper used GPT-4o, so language metrics reflect the
     selected provider.

5. **`agents/system_prompt.py`, `utils/equ_ranking.py`, `utils/chat_utils.py`**
   — the discriminator now returns plain-text `correct` or `incorrect`.
   - `chat_with_gpt4v` has a `json_mode` flag: the generator requests JSON,
     and the discriminator requests one word.
   - Reason: DeepSeek sometimes used the key `correct`, which caused
     `KeyError: 'correctness'`.
   - The binary decision semantics remain unchanged; the response transport
     format changed.

6. **`main_vis.py`, `main_vln.py`** — added a GUI-close stop for demos.
   - Closing the Open3D window sets a flag; the navigation loop stops at the
     next agent step.
   - GUI runs wait for the Open3D thread, flush logs, then exit through
     `os._exit`, which avoids Open3D's wrong-thread shutdown panic.
   - Reason: release the process and GPU promptly when a demo ends.

7. **`demo.py`** — interactive launcher for the three demo modes.
   - ObjectNav mode (reference), Language mode: CLIP-based, and Language mode: VLM.
   - Always requests GUI mode.
   - Asks for episode count in all modes.
   - Asks for a model only in VLM mode, using the dynamic `VLN_MODELS` list.
   - CLIP mode requires no API key.
   - Reason: simplify local GUI demos; it calls the existing entrypoints.

8. **`utils/vis_gui.py`** — user-approved GUI-thread/callback compatibility fix.
   - Episode-change scene clearing and FPS/info widget updates now execute on
     the GUI event-loop thread instead of the `UpdateMain` worker.
   - Deferred callbacks bind their frame/candidate data and episode identity;
     image-save callbacks use the identity of the request that produced them.
   - Reason: address concrete threading defects after recurring native
     `FEngine::loop` segmentation faults during GUI evaluation. The exact
     native crash cause is not yet proven.
   - Navigation, evaluation, manual-query/pause controls, image composition,
     and the shared `Vis-<id>.jpeg` filename convention are preserved. The
     candidate-image filename collision was still pending at this revision;
     change 11 subsequently corrected it.
   - Validation (2026-10-25): delayed-frame/episode-boundary GUI checks,
     six synthetic candidate composites, and query/pause/resume callbacks
     passed. A three-episode DeepSeek `vlm_game` GUI smoke run completed
     in 17m37s with 1,095 frame updates, 19/19 candidate renders, no thread
     assertion failures or new kernel crash records, and clean GUI shutdown.
     This short test does not establish multi-hour stability.

9. **`main_vln.py`, `agents/vlnav_agent_vlm.py`, `utils/equ_ranking.py`,
   `utils/chat_utils.py`, `utils/equ_trace.py`** — user-approved structured
   equilibrium tracing for single-process `vlm_game` runs.
   - Unique run manifests record exact prompts, non-secret effective API
     configuration, task configuration, and source identity. Episode JSONL
     files contain parsing, search metadata, raw logical sample returns/errors,
     parsed values, aggregates, policy mappings, initialization and updates
     125/250/375/500, returned decisions/errors, and episode outcomes.
   - Images are archived from the exact buffer encoded for each request, using
     unique search/input/candidate filenames and SHA-256 hashes. Existing file
     reads, image dimensions, candidate ordering, and render behavior are not
     changed. Unexpected images are preserved and flagged.
   - Legacy raw response logging moves into the structured trace when tracing
     is active; `eval.log` retains metrics and short trace references. Events
     are flushed immediately so partial traces survive native process crashes.
   - At this stage, the known Discriminator accumulation defect, fixed-five denominator,
     reject-all branches, and numerical anomalies are preserved. Each raw
     response is traced once, with the actual reused scoring pools recorded
     separately. Nonfinite numbers use explicit JSON string representations.
   - This observability change preceded the separate aggregation correction in entry 12.
   - Validation (2026-10-25): ten offline tests passed, plus five comparisons
     against pre-change source with identical mocked requests, policy arrays,
     and returned candidates (including the accumulation/NaN case). A real
     three-episode DeepSeek GUI run completed in 17m11s: ten searches, 19 archived
     inputs, 60 logical sample returns, and ten policy snapshots. An independent
     audit matched every request's ordered input hashes to the archives and
     recomputed all checkpoints from the recorded inputs. There were no thread
     failures or new kernel crash records, and shutdown released the GPU.
   - That smoke archive confirmed 15/19 actual VLM input buffers were 640x800
     rather than 960x480; all were retained and flagged. This validates the
     archive's fidelity, not candidate-image correctness or multi-hour stability.

10. **`utils/equ_report.py`, `utils/equ_trace.py`** — user-approved plain-text
    episode reports derived from the structured trace.
    - Each closed episode automatically receives `equilibrium.txt` alongside
      `equilibrium.jsonl`. `python -m utils.equ_report <archive-or-jsonl-path>`
      regenerates reports for existing or interrupted archives without API calls.
    - Reports show raw opinions, actual candidate mappings and scoring pools,
      initialization inputs, all four policy columns at each checkpoint, episode
      outcomes, image references, and numerical/input anomalies. Output is ASCII,
      100-column wrapped, with display rounding only; JSONL/images are unchanged.
    - Truncated final records and missing completion events are explicitly marked.
      Earlier malformed records are not silently skipped. Reporting failures do
      not abort navigation or change metrics. The later candidate-image and
      Discriminator corrections are recorded separately in entries 11 and 12.
    - Validation (2026-10-25): all 20 offline trace/report tests passed. Reports
      were generated for all three smoke-run episodes (ten searches); SHA-256
      checks confirmed all 23 original manifest/JSONL/image files were unchanged.

11. **`utils/candidate_images.py`, `utils/vis_gui.py`,
    `agents/vlnav_agent_vlm.py`, `main_vln.py`, `utils/equ_trace.py`,
    `utils/equ_report.py`** — user-approved candidate-image delivery correction.
    - Ordinary GUI screenshots use `frames/Frame-<frame>.jpeg`. Candidate
      composites use unique run/episode/search/candidate paths. The navigation
      agent no longer accepts a nonempty shared `Vis-<id>.jpeg` as readiness.
    - A dedicated single-process completion channel links each render request
      to its candidate and selected first-view buffer hash. JPEG publication is
      atomic; the consumer validates identity, SHA-256, decodability, JPEG format,
      and 960x480 dimensions, then archives/encodes that same read buffer.
    - GUI callbacks never wait for completion; only worker/navigation threads
      wait. Pending waits are cancelled on GUI closure, bounded by a 120-second
      per-image timeout, and failed explicitly on renderer/publication errors.
      Render errors are not converted into STOP actions or benchmark failures;
      traces mark the interrupted/error episode and error shutdown closes the GUI.
    - Composite layout, camera, bounding boxes, prompts, candidate ordering,
      sample counts, navigation thresholds, metrics, and Discriminator calculations
      are preserved. Old archives are not rewritten; corrected-image results must
      be distinguished from uncorrected reference runs. The Discriminator
      aggregation bug remained deferred at this stage; its later approval and
      correction are recorded in entry 12.
    - Validation (2026-10-25): 33 offline tests passed. Actual GUI checks exercised
      12 frame updates, nine candidate composites across three episodes, repeated
      IDs/searches, delayed callbacks, stale files, manual query/pause/resume, and
      cancellation of a pending render on GUI close. Composite JPEG bytes matched
      the pre-fix composition for identical synthetic inputs (IDs 0/15/24).
    - A three-episode DeepSeek `vlm_game` GUI smoke run completed in 9m28s:
      nine searches, 13/13 fresh 960x480 inputs, 80 logical VLM samples, and 35
      policy snapshots. An independent ledger matched all render identities and
      first-view hashes through publication, trace archives, and API input hashes;
      all checkpoints were recomputed successfully. No render/thread failures or
      new kernel crash records occurred; shutdown released the GPU. Success was
      0/3 (SPL 0, DTG 7.748); this validates delivery, not benchmark performance
      or multi-hour GUI stability. Historical artifacts: `/tmp/opencode/candidate-image-vlm-smoke/` (currently unavailable; see the evidence-location note above).

12. **`utils/equ_ranking.py`, `utils/equ_trace.py`** — explicitly user-approved
    Discriminator aggregation correction affecting the equilibrium algorithm.
    - Reset each eligible candidate's `responses` and `response_ids` immediately
      before its queries. A=4/5 and B=2/5 now produce 0.8 and 0.4, rather than
      0.8 and 1.2. See the detailed discriminator explanation above for cause,
      reproduction validity, and the minimal fix.
    - Five-sample requests, images/prompts, bias, normalization/update equations,
      candidate mapping, reject-all branches, failed-query denominator behavior,
      and invalid-response handling are preserved. There is no clamping.
    - New manifests use `candidate_local_responses_v1`. Existing archives and
      their aggregation labels are untouched; synthetic legacy report fixtures
      retain accumulated scoring pools, negative policy entries, and NaNs.
    - Offline comparisons against pre-correction source verify identical API
      request contents/counts and unchanged policies/decisions for unaffected
      branches. The affected NaN case becomes finite, and a valid-frequency
      example changes its selected candidate from B to A as intended.
    - Validation (2026-10-25): all 41 offline tests passed. The GPT-4o-mini
      GUI smoke run completed three episodes in 12m55s: seven searches, 13/13
      validated 960x480 inputs, 60 logical samples, and 25 policy snapshots.
      An independent audit verified candidate-local scoring pools, replayed all
      checkpoints, and matched render/archive/API image hashes. All supplied
      policy diagnostics were finite/nonnegative/normalized. The five initialized
      live searches had one discriminated candidate each; multi-candidate
      correctness is directly covered by the offline two/three-candidate cases.
      There were no thread failures or new kernel crash records; shutdown
      released the GPU. Success was 1/3 (SPL 0.089, DTG 4.316), not benchmark
      evidence of an improvement. Historical artifacts: `/tmp/opencode/discriminator-corrected-mini-smoke/` (currently unavailable; see the evidence-location note above).

13. **`utils/vis.py`** — Supervision annotation API compatibility update.
    - Replace deprecated `BoxAnnotator` with `BoundingBoxAnnotator` and
      `LabelAnnotator`, and `ColorPalette.default()` with `ColorPalette.DEFAULT`.
      The installed Supervision remains 0.21.0; no dependency upgrade is needed.
    - Preserve black text, scale 0.3, text thickness 1, padding 2, box thickness 2,
      palette, mask drawing, and per-detection box/label ordering. Annotation
      image pixels match the old implementation across empty/masked/unmasked,
      overlapping, border, custom-colour, and instance-colour tests. Input images
      and detection arrays are not mutated; repeated calls emit no deprecation
      warnings. This change is visualization compatibility, not a detector,
      VLM-input, or navigation algorithm change.
    - The combined GPT-4o-mini GUI smoke test processed 865 frame updates with
      zero `BoxAnnotator` / `ColorPalette.default()` deprecation warnings.

14. **`utils/chat_utils.py`, `utils/equ_ranking.py`, `utils/equ_trace.py`,
    `utils/equ_report.py`, `main_vln.py`** — API reliability correction, deliberately
    retaining the existing Requests and OpenAI SDK clients.
    - Configure connect/read-inactivity timeouts of 10/90 seconds on supported
      parsing and vision requests. Use at most three application attempts with
      2/4-second backoff and capped Retry-After handling (30 seconds). Malformed
      model syntax/empty content and outer-response errors consume the budget;
      permanent HTTP/configuration/SDK-authentication errors fail without retries.
    - Check HTTP status and choices/message/content structure before extraction.
      Record per-attempt role/sample identity, elapsed time, HTTP status, content
      type, request ID, failure category, and bounded sanitized previews. Never
      record authorization headers or keys. Attempt callbacks use serialized
      JSONL writes; terminal reports expose the diagnostics too.
    - Adopt the recommended `abort_test_no_partial_batch` failure policy: after
      a required sample error, do not produce an aggregate or navigation result
      from a partial batch. The affected episode is marked error/incomplete;
      completed episode results remain intact. Peer calls skip further retries
      after a batch failure/interrupt; current HTTP calls still rely on their
      socket timeouts. Error/interrupt cleanup requests GUI closure.
    - This is not an async transport rewrite, a hard wall-clock deadline, or
      forcible cancellation of in-flight threads. A server trickling bytes can
      extend a read timeout; the SDK can also perform internal connection retries.
      Logs therefore count application attempts, not all billed HTTP requests.
      The legacy unused local-Llama helper remains outside this supported path.
    - Prompts, images, model options, output caps, temperatures, five logical
      samples, and equilibrium calculations for complete batches are preserved.
      Historical partial-batch/NaN traces and their denominators are not rewritten.
    - Validation (2026-10-26): all 61 offline tests passed, including a local
      stalled HTTP server, retry exhaustion, permanent errors, sanitized diagnostics,
      interrupt cleanup, and exclusion of incomplete episodes from metrics. A
      synthetic fatal-error GUI probe exhausted three empty-response attempts for
      one required sample: no aggregate, policy, or episode metrics were produced,
      and the GUI closed automatically.
    - A three-episode GPT-4o-mini GUI smoke run completed in 8m42s with one
      deliberately injected empty HTTP 200 response. Its retry succeeded and
      contributed exactly one logical vote; the Generator denominator remained
      five. The run recorded six searches, 11 validated 960x480 inputs, 55 logical
      vision returns, 15 policy snapshots, and 59 application attempts (three
      parsing and 56 vision, including the injected failure). One search
      discriminated two candidates. Independent audits matched ordered API image
      hashes, candidate-local scoring pools, and replayed policy checkpoints.
      All 11 candidate renders completed across 610 frame updates; shutdown was
      clean, with no thread failures or matching new kernel crash records, and
      the GPU was released. Success was 1/3, SPL 0.089, DTG 3.415; this short,
      fault-injected run validates reliability rather than benchmark performance.
      Persistent artifacts, source snapshots, and validation reports are under
      `saved_sessions/smoke_tests/20261026-http-timeout-mini-smoke/` and
      `saved_sessions/smoke_tests/20261026-http-timeout-fatal-gui-check/`.

15. **`utils/candidate_renderer.py`, `utils/candidate_images.py`,
    `utils/vis_gui.py`, `agents/vlnav_agent_vlm.py`, `main_vln.py`,
    `utils/equ_trace.py`, `utils/equ_report.py`** — user-approved headless VLM path.
    - Approval (2026-10-26) followed a native `FEngine::loop` crash after about
      3h13m: 51/100 episodes completed and episode 52 incomplete. Before source
      edits, the original run, all 405 exact VLM inputs, hash-matched source,
      evaluation log and kernel evidence were copied and verified under
      `saved_sessions/20261026-gpt-4o-mini-vlm-game-51of100-render-crashed/`.
    - `-v 0` uses one synchronous, persistent Open3D offscreen renderer on the
      navigation thread. GUI imports/window/update thread are only entered for
      `-v 1`. No renderer subprocess, automatic restart, or transport redesign.
    - The fixed 640x800 viewport was captured from the unmodified local GUI
      (1280x800 window, theme font size 32). Preserve FOV 90, near plane 0.1,
      far-plane calculation using original fixed [-5,5]^3 scene bounds, camera
      offset [-0.5,4,0], up [0,1,0], full point cloud, materials and red bbox.
      Shared composition preserves the first-view image, label and JPEG settings.
    - Reuse request-specific paths, first-view hashes, atomic publication,
      completion validation and exact-buffer trace/API archival. Backend and
      camera provenance are separate from the unchanged delivery protocol and
      `candidate_local_responses_v1` aggregation label. Prompts, model settings,
      navigation logic, thresholds, samples, equilibrium and metrics are preserved.
    - Offscreen still uses Filament and is not crash isolation. Its synchronous
      native call has no enforceable Python deadline; manifests record null for
      the GUI completion timeout in this mode. Python render/publication errors
      abort without fabricated actions or incomplete-episode metrics.
    - Initial validation: DISPLAY-unset EGL rendering passed; six actual renders
      matched GUI view/projection matrices exactly across IDs 0/15/24, repeated
      renders and translated geometry. Raw cross-backend mean pixel differences
      were 0.466-0.471/255 (99th percentile 2), versus GUI repeat differences
      0.460-0.461/255. Sparse edge differences remain; rendered images are not
      byte-identical. Shared compositor pixels and JPEG bytes exactly matched
      pre-change source for IDs 0/15/24. All 68 offline tests passed.
    - A real three-episode GPT-4o-mini smoke completed in 6m40s with DISPLAY unset
      and no GUI module loaded. Six searches produced 11 validated 960x480 inputs,
      45 logical vision returns, 15 policy snapshots and 48 application attempts,
      with no retries or rendering errors. An independent audit matched ordered
      API input hashes and replayed candidate-local scoring/policy checkpoints.
      Exit was clean with no matching new kernel crash records. Success was 1/3,
      SPL 0.089, DTG 3.415; this is smoke validation, not benchmark evidence.
      Artifacts: `saved_sessions/smoke_tests/20261026-headless-mini-smoke/`.
    - The four-hour renderer-only test was cancelled at the user's explicit
      request on 2026-10-26, after 186 renders / 185.58 seconds. SIGINT stopped
      PID 866790; exit code -2 and the script's generic error status reflect
      cancellation, not an observed renderer failure. Evidence and an explicit
      `CANCELLED.md` note remain under
      `saved_sessions/smoke_tests/20261026-headless-render-endurance/`.
      No four-hour stability result was obtained. The user instead approved a
      real 100-episode GPT-4o-mini headless navigation run, archived separately
      under `saved_sessions/20261026-gpt-4o-mini-vlm-game-headless-100ep/`.
      That run stopped after 70/100 completed episodes in 3h11m49s (exit code 1)
      on episode 71's first step: the parsing response contained
      `landmark: ["table", "flower vase", []]`, and GroundingDINO's string join
      raised TypeError. This was a parsing-output structure error, not a native
      renderer crash; episode 71 has null metrics and is excluded. Results for
      completed episodes: 22/70 successes, SPL 0.108471, DTG 3.543361. Final archive
      summary, failure evidence, documentation snapshots and SHA-256 inventory are
      retained in that session directory. No parser behavior was changed by the
      archival task; the 100-episode completion target remains unmet.

16. **`utils/chat_utils.py`, `tests/test_chat_utils.py`** — user-approved parsing
    structure validation and bounded regeneration (2026-10-26).
    - Approval followed the archived headless run's episode-71 failure: a nested
      empty list among landmarks reached GroundingDINO's string join. The response
      was syntactically parseable; this corrects output acceptance/retry behavior,
      not JSON syntax or navigation mathematics.
    - In the existing shared request loop, retain `ast.literal_eval` and require,
      only for the instruction-parsing role, a dictionary with a string `target`
      and an optional list of string `landmark` entries. Missing landmarks and an
      empty list remain valid; extra fields and empty strings retain their prior
      acceptance. Valid response text is returned unchanged.
    - Reject an invalid shape as `parsing_structure` with the existing bounded,
      sanitized preview. Retry the identical request within the same three-attempt
      budget and 2/4-second backoff, then raise the existing `APIRequestError` on
      exhaustion. No extra retry loop, prompt change, schema framework, coercion,
      flattening, landmark removal, Generator/Discriminator change or parser swap.
    - Manifests record `parsing_output_validation` as
      `target_string_optional_landmark_string_list_v1`, separately from unchanged
      image-delivery, aggregation, transport and failure-policy labels.
    - Validation: all 72 offline tests passed, including the exact archived raw
      response followed by a valid response, unchanged request kwargs, invalid
      shape exhaustion, optional/empty landmark lists, and Generator scope.
      An online recovery probe replayed the exact malformed response for attempt
      1; attempt 2 used the real GPT-4o-mini provider with identical archived
      prompt/settings and returned valid string landmarks. This injected replay
      is explicitly labelled, not counted as an actual provider failure.
    - A separate natural-provider, single-episode headless replay of source episode
      71 (dataset ID 6, HM3D `00810-CrMo8WxCyVb`) completed successfully: 193 steps,
      success 1.0, SPL 0.120460, DTG 0.111536 m; exit code 0 in 2m10s including
      startup. The parsing call returned valid string landmarks on attempt 1.
      This episode made no candidate searches/vision calls, so it does not newly
      validate rendering or equilibrium branches. The harness selects the existing
      episode without changing production configs/data; instruction, scene, task
      parameters, model requests and start pose are verified. It is a cold-start
      targeted test, not a replay of the preceding 70 episodes or stopped benchmark.
      Evidence: `saved_sessions/smoke_tests/20261026-parsing-structure-episode71/`.
      The initial navigation harness preflight compared the old manifest's first
      scene with the selected episode; it stopped before navigation-phase API calls.
      That harness-only mismatch is retained, and the separate navigation phase
      compares unchanged task parameters while verifying the actual selected scene.

17. **`demo.py`** — user-approved headless/GUI launcher selection (2026-10-26).
    - Ask for Headless (no window) or GUI (Open3D) before navigation mode, episode
      count or model. Apply the selected `-v 0` / `-v 1` to ObjectNav, CLIP and VLM
      launches and display the choice in the final summary.
    - Require DISPLAY only for GUI mode. Keep existing entrypoints, model registry
      and private environment handling, count/back/quit flows and full-run
      confirmation. No navigation, rendering backend, manual-query/pause controls,
      evaluation logic, prompts or API payloads changed.
    - Validation: 13 offline launcher checks using scripted inputs and mocked
      subprocess launches passed. They cover all six mode/display combinations,
      all three GUI-without-DISPLAY rejections, initial quit/invalid choices,
      custom count and the existing default-no full-run confirmation. Headless
      checks ran without DISPLAY; no navigation process or API call was started.
      These verify launcher routing, not new simulator/render endurance results.

18. **`utils/equ_ranking.py`, `main_vln.py`, `agents/vlnav_agent_vlm.py`,
    `utils/equ_trace.py`, `utils/equ_report.py`, `tests/test_generator_baselines.py`**
    — user-approved non-destructive Generator/Ranking baseline correction and tracing
    (2026-10-27), after the full DeepSeek Game measurement was archived.
    - Preserve existing flags/default: `vlm` uses one Generator sample,
      `vlm_rank` uses five, `vlm_game` remains default and calls the existing
      equilibrium path. This follows the released code's sample counts and the
      paper's five-sample Ranking description, not a new ranking prompt.
    - Correct the inherited baseline wrapper: unpack the helper's ID/probability
      pair and return the actual object ID, not an index into that pair or array.
      Reject (`-1`) continues exploration. Ranking selects the plurality winner;
      first-max ties follow offered candidate order, with rejection appended last.
      A real candidate therefore wins a tie against rejection. No random tie RNG,
      majority threshold, extra voting queries or Discriminator was added.
    - Baseline-only `validate_ids=True` requires a string object_id from the
      offered IDs or `-1`. Invalid IDs raise before aggregation; they are not
      dropped/coerced/relabelled as rejection. This semantic check does not add a
      retry budget. The shared helper defaults validation off, preserving Game's
      existing model-output interpretation and approved API/parse retry behavior.
    - Reuse existing trace/image publication infrastructure for `vlm` and
      `vlm_rank`: raw responses, API attempts, sample counts, vote frequencies,
      tied options, selected IDs and episode metrics. Manifests identify
      `generator_single_sample_v1` / `generator_plurality_5_v1`, sample count and
      `discriminator_enabled=false`; Discriminator aggregation is `not_applicable`.
      Baseline reports explicitly state zero equilibrium updates, not missing
      policy data. Existing Game aggregation labels/math remain unchanged.
    - All 79 offline tests passed. Seven new tests cover actual non-consecutive
      IDs, one/five sample counts, plurality, ties including reject, invalid IDs,
      required-sample failure, trace closure and baseline reports. Five controlled
      before/after Game cases match the sealed full-run source in request roles,
      image order/counts, decisions/errors and checkpoint arrays. Game search,
      Discriminator and policy-update/normalization method ASTs are unchanged.
    - Both natural DeepSeek headless smoke runs completed 3/3 episodes. Generator:
      8m49s, 7 searches, 13 images, 7 logical Generator samples, 11 application
      attempts (one retry), 2/3 successes, SPL 0.090145, DTG 3.052540. Ranking:
      6m24s, 6 searches, 11 images, 30 logical Generator samples, 33 application
      attempts (no retries), 1/3 successes, SPL 0.088991, DTG 3.414690. Both exit 0;
      zero Discriminator queries/equilibrium updates. Independent checks verify
      image/source hashes and recompute every baseline decision from votes.
      These are functional checks, not evidence of benchmark superiority.
    - Evidence: `saved_sessions/smoke_tests/20261027-baseline-method-validation/`,
      `20261027-deepseek-generator-smoke/` and `20261027-deepseek-ranking-smoke/`
      under the same smoke_tests directory. No detector, prompt, rendering camera,
      model stack, navigation threshold, dataset or evaluation change was made.
    - Later full Generator attempt (same day, commit `1eb0267`): stopped after
      5/316 completed episodes at episode 6/search 4. Offered IDs `[0,3,18,19,21]`,
      returned string ID `"4"` (HTTP 200). The baseline-only fail-fast membership
      guard raised outside the API retry loop; no aggregation or semantic-ID
      regeneration occurred. This was an explicit validation stop, not a native
      crash. The short smokes did not encounter this response; offline invalid-ID
      tests intentionally verified this abort policy. Evidence remains in
      `saved_sessions/20261027-deepseek-v4.1-flash-generator-headless-316ep/`.
      A bounded regeneration follow-up is proposed separately; no failed sample
      was repaired/relabelled and no automatic restart was performed.

19. **`utils/chat_utils.py`, `utils/equ_ranking.py`, `main_vln.py`,
    `tests/test_chat_utils.py`, `tests/test_generator_baselines.py`** — user-approved
    bounded candidate-ID regeneration for Generator/Ranking (2026-10-27).
    - The first full Generator run stopped after 5/316 episodes because the model
      returned string `"4"` for offered `[0,3,18,19,21]`. Change 18's semantic guard
      was outside the API retry loop, so it failed fast despite HTTP 200 and valid
      syntax. That stopped-run archive remains unchanged and separately labelled.
    - Baseline batches now pass an optional `allowed_ids` set to the existing
      vision client. Within its existing JSON validation stage, an invalid shape,
      missing/non-string ID or unoffered ID raises retryable `candidate_id` with
      the original bounded sanitized preview. The same prompt/images/options are
      retried within the existing three total application attempts and 2/4-second
      backoff. Transport, syntax and semantic errors share one budget. No new
      loop, coercion, ordinal guessing, vote dropping or rejection substitution.
    - Only final accepted responses enter the one-sample Generator / five-sample
      Ranking aggregation. Exhaustion aborts without partial metrics/aggregation.
      The post-return membership guard remains as a defensive check. Game does
      not opt in (`allowed_ids=None`), so its request/interpretation/math behavior
      remains unchanged. Manifests label baseline policy
      `offered_string_id_bounded_retry_v1`; Game records `not_applicable`.
    - All 84 offline tests passed, including exact invalid-ID recovery, unchanged
      payload, invalid-type/shape exhaustion, shared timeout/ID budget, legacy
      Game acceptance and six HTTP attempts still yielding only five Ranking votes.
      Five before/after Game comparisons still match requests, decisions/errors
      and checkpoint arrays; Game core method ASTs remain unchanged.
    - Static online smoke reconstructed the exact failed five-image request from
      archived JPEG buffers, original instruction/prompt, direct DeepSeek model
      settings and timeouts. Attempt 1 explicitly replayed the archived `"4"`
      response; attempt 2 called the real provider with unchanged payload and
      returned ID 18. Exactly one logical Generator response/vote was aggregated.
      There was no Habitat/navigation/rendering or Discriminator/equilibrium work.
      The injection is labelled, not attributed as a new provider failure.
      Evidence: `saved_sessions/smoke_tests/20261027-baseline-id-retry-static/`.
    - The user authorized restarting the same full headless DeepSeek Generator
      configuration after commit/push, with manual `sleep 1800`/check monitoring
      instead of hourly checks. It starts from the normal iterator beginning,
      without skipping episodes or merging the failed attempt's metrics.

20. **Isolated Jev backend and Jev-only frozen instruction parsing** — approved
    2026-10-28 after the completed DeepSeek benchmarks; implemented locally.
    - Add `typesafe/jev-1.14` through OpenRouter Decisions, not `jev-router` or chat
      completions. `utils/jev_client.py` constructs native state/questions from
      original instructions, real candidate IDs and exact JPEG buffers. Omit chat
      generation options. Use the existing bounded transport/retry machinery;
      malformed typed outputs share the same budget, with no partial aggregation.
    - User retained names/flags VLN-generator (`vlm`), VLN-ranking (`vlm_rank`),
      VLN-game (`vlm_game`). Jev Generator uses one Choice; Ranking averages five
      complete Choice distributions with equal weights, rather than hard-vote
      plurality. Candidate-order ties put rejection last. Confidence is retained
      as metadata, not used for weighting/thresholds; no dedicated cost tracking.
    - Jev Game averages five Choices over all offered candidates plus rejection,
      and five candidate-local Nouls per real candidate. Multiple candidates use
      five all-image none-match Nouls; one candidate derives rejection by complement.
      Match probabilities are not normalized across candidates.
    - User explicitly instructed arrays instead of the original equilibrium
      dictionaries. `utils/jev_ranking.py` uses G 2×K / D K×2, `[g;1-g]` and
      `[d,1-d]`, bias 0.05, row-normalized initial policies, zero Q arrays, 500
      accumulated opponent-policy updates weighted `1/(2t)`, lambdas/etas 0.1,
      simultaneous previous-policy inputs and row-wise stable softmax. Select
      final G correct-row argmax. No sampled-support filtering/early exit,
      column initialization, running-average payoff replacement or 5,000 updates.
    - Jev-only parsing uses `utils/instruction_lookup.py`, the preparation tool
      `utils/prepare_jev_lookup.py`, and `configs/jev_instruction_parses.v1.json`
      with its checksum sidecar. All 316 unique exact instructions were parsed
      by direct DeepSeek Flash, unchanged Parsing_prompt, temperature 0.5,
      max_tokens 512, disabled thinking, original shape validation/retry budget.
      All 316 application attempts returned accepted parses without retry; raw
      provider responses and journals are preserved. No key normalization,
      repeated sampling to choose a parse or manual rewriting. Full coverage,
      dataset fingerprints and checksum are mandatory; no live fallback.
      Structural validation does not prove semantic accuracy. Only perception/
      target similarity consumes parsed phrases; Jev gets the full instruction.
    - Preserve other models' live parsing, requests, aggregation and equilibrium
      methods. Registry selection appends Jev without replacing private entries;
      switching resets effective model/key/options. No protected GUI/pause,
      arguments, requirements, dataset episodes or navigation thresholds changed.
    - Traces/reports label native probabilities, frozen-table identity and array
      axes/checkpoints; record raw typed responses, validated pooled vectors and
      Q/policy snapshots. Source identity includes the new implementation files.
      Strict typed validation requires complete finite unit-interval vectors;
      mass deviations at most 1e-6 are normalized for roundoff, larger deviations
      retry. Missing options are never zero-filled and no values are clamped.
    - Verification: all 98 offline tests passed, including solver parity against
      the existing dictionary equations at every recorded checkpoint (K=2/4/8),
      full-support/rejection branches, probability-vs-vote distinction, lookup
      integrity, native HTTP failure handling and model switching. Five sealed-
      reference chat Game comparisons match requests, errors and checkpoint
      arrays; its core method ASTs remain unchanged. Each Jev method completed a
      three-episode natural headless smoke with actual exit code 0, nine searches
      and 13 exact input images. All had 0/3 successes, SPL 0, DTG 7.748133 m;
      these verify operation, not accuracy. All means/selections and 45 Game policy
      checkpoints were independently replayed; 184 application attempts, no retry
      or terminal failure. Detailed results/caveats are maintained in the status page.
    - Evidence: `saved_sessions/smoke_tests/20261028-jev-lookup-generation/`,
      `20261028-jev-backend-validation/` and the preceding exact-input
      `20261028-jev-repeatability/` probe (all under smoke_tests). Table v1 SHA-256:
      `d4abcaaf62f8cdf107e9490a71b60af80446607839454f9d40b7b4a6ac83812a`.

21. **`demo.py` VLM method menu** — user-requested launcher selection (2026-10-28).
    - After choosing VLM navigation and episode count, select VLN-generator
      (`vlm`), VLN-ranking (`vlm_rank`), or VLN-game (`vlm_game`), then the model.
      Enter defaults to Game; Back returns to episode-count selection.
    - Pass the chosen `--vln_mode` explicitly, display the method in the launch
      summary and print its matching evaluation-log path. Applies to Jev and
      existing chat models. ObjectNav/CLIP flows and full-run confirmation remain
      as before; no inference, parsing, navigation or equilibrium behavior changes.
    - Verification: 21 offline mocked launcher cases passed, covering all methods,
      Enter default, Jev/DeepSeek, headless/GUI, Back, invalid input, ObjectNav/CLIP
      menu skipping, matching summary/log paths and default-no full-run confirmation.
      All 98 existing offline tests passed. No navigation process or API call.
      Evidence: `saved_sessions/smoke_tests/20261028-demo-method-selection/`.

#### Data compatibility changes

The data were downloaded from [Github](https://github.com/matterport/habitat-matterport-3dresearch).

Original compressed data filenames:

1. hm3d-val-habitat.tar                 3.3G
2. hm3d-val-semantic-configs-v0.1.tar   20K

The original tarballs are archived at
`$HOME/Documents/VLN-Game-hm3d/archives/`; only the scene-data directory is
symlinked into the repo.

However, because the downloaded files have newer episode JSON format, the
following data compatibility changes were made:

- Removed `scene_dataset_config` and `additional_obj_config_paths` from the
  episode JSON files under `data/datasets/objectgoal_hm3d/` and
  `data/datasets/vlobjectnav_hm3d_v4/`.
- Reason: Habitat 0.2.1's episode classes reject these newer fields with
  `TypeError: ... unexpected keyword argument 'scene_dataset_config'`.
- Goals, viewpoints, instructions, start poses, rotations, scene IDs, and
  metrics remain unchanged.
- The extracted ObjectNav directory was renamed from `objectnav_hm3d_v1` to
  `objectgoal_hm3d` to match the repo config.

#### Environment-only changes

These changes live in the conda environment:

- `numpy==1.23.5` because Habitat 0.2.1 uses the removed `np.float` alias.
- `numpy-quaternion==2023.0.4` to stay compatible with that NumPy version.
- `faiss-cpu==1.7.4` because `utils/mapping.py` imports `faiss`.
- `supervision==0.21.0` because `utils/detection_segmentation.py` imports it.
- `transformers==4.36.2` and matching tokenizers/huggingface-hub/safetensors
  because newer Transformers disables the installed PyTorch version.
- CMake 3.25 because CMake 4 fails habitat-sim's policy checks.
- GroundingDINO built with `--no-build-isolation` and the conda CUDA toolkit
  to avoid a PyTorch/CUDA mismatch.
- MobileSAM because the 6 GB VRAM limit makes SAM ViT-H impractical.

## Earlier validation notes (historical)

The following observations predate the later fixes and long runs above. They
are retained for provenance, not as active tasks, current provider-availability
checks or outstanding holds. The earlier baseline-first investigation has a
completed CLIP archive; see [current reproduction status](REPRODUCTION_STATUS.md).

ObjectNav GUI smoke run: 5 episodes, 4 successes, SPL 0.235. Episode 1
failed; episodes 2-5 succeeded. The Open3D GUI appeared and the manual query
box worked.

API smoke checks passed for the four configured models: GPT-4o, GPT-4o
mini, GPT-6 Luna, and DeepSeek V4.1 Flash. Each returned valid text JSON,
generator JSON, and discriminator plain text at that time. GPT-4o required retries because
OpenRouter returned empty content on some attempts.

A 6-episode DeepSeek language batch reached the generator/equilibrium path in
all five new episodes; episode 5 accepted a candidate and succeeded (success
0.200, SPL 0.010). The batch predates the discriminator plain-text change and
the model registry. The old note proposed a rerun; it is not a current instruction.
Candidate-level behavior from that batch is suspect because candidate-image
integrity was unresolved at that revision (subsequently corrected in change 11).

Language metrics reflect the selected provider; the paper used GPT-4o.

The GUI-close stop was tested on both entrypoints; both processes exited and
released the GPU. The interactive launcher now exposes the three demo modes;
ObjectNav launch, CLIP command construction without an API key, and VLM model
selection were tested. The old note said the CLIP baseline was pending; it later
completed 316/316 episodes and is recorded in the status page and session archive.
