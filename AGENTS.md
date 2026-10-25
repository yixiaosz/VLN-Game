# AGENTS.md

## Repo Purpose

This repository is the VLN-Game reference implementation: a zero-shot
object-goal and language-goal navigation framework using GroundingDINO,
MobileSAM, CLIP, a 3D object-centric map, frontier exploration, and
game-theoretic VLM ranking.

The personal purpose of this local checkout is to reproduce the original
paper's algorithm and probe its performance in simulated Habitat/HM3D
environments. The target is a faithful reproduction. Refactors and redesigns
are outside scope.

"Working" means:

- The environment, models, datasets, and simulator load successfully.
- The Open3D GUI appears and shows navigation/mapping.
- A 3-5 episode smoke run completes with sane metrics.
- The full 2,000-episode ObjectNav / 316-episode language benchmarks are
  only run when explicitly requested.

### Editing Rules for Agents

1. **Fidelity first.** Do not change the algorithm, hyperparameters,
   evaluation logic, metrics, observation space, action space, or dataset
   semantics to improve results.
2. **Only fix blockers.** Code changes are allowed only when needed to make
   the original code install, import, load data, or run on this machine.
   Examples: incompatible dataset fields, dead imports that fail, hardcoded
   debug bypasses that prevent episodes from running.
3. **Discuss behavioral changes first.** If a change affects navigation
   behavior, evaluation, metrics, or the GUI interaction model, stop and ask
   the user before editing.
4. **Keep diffs minimal.** Do not refactor, rename, reformat, or clean up
   unrelated code. Do not replace the paper's model stack or pipeline.
5. **Do not tune for the benchmark.** Never adjust thresholds, prompts,
   episode selection, or configs merely to raise success/SPL.
6. **Do not run the full benchmark by default.** Use a 3-5 episode smoke run
   unless the user explicitly asks for the full evaluation.
7. **No sudo.** Do not run `sudo` commands. List any required privileged
   command for the user to run.
8. **No commits.** Do not commit, amend, push, or create PRs unless the user
   explicitly requests it.
9. **No secrets.** Never write API keys, HM3D credentials, tokens, or
   passwords into the repo, logs, or this file.
10. **Preserve original files.** Leave `requirements.txt`, `arguments.py`,
    `utils/vis_gui.py`, and the GUI manual-query behavior unchanged unless a
    documented blocker requires otherwise.
11. **Document approved persistent changes.** If the user approves a
    permanent source or data compatibility change, record it under
    "Changes From the Original Paper Repo" below.
12. **Do not delete user data or tarballs** without explicit permission.
13. **Freeze during baseline measurement.** While establishing the original
    CLIP-only baseline, do not change program logic. The candidate-image issue
    and structured trace are documented pending work; implement them only
    after explicit user approval.
14. **Preserve the original README section.** In `README.md`, everything below
    the line `(Below this line is the original README of the [VLN-game](https://github.com/ybgdgh/VLN-Game) code)`
    is the original paper-code README. Do not edit it.
15. **Preserve aggregation provenance.** The user explicitly lifted the earlier
    hold and approved the Discriminator correction on 2026-10-25, before a
    completed 100-episode GPT-4o reference test. New runs use candidate-local
    response batches, labelled `candidate_local_responses_v1`. Preserve old
    uncorrected archives and their labels. The later API reliability change
    (entry 14) stops a test after a required sample fails rather than producing
    partial-batch frequencies. Preserve historical failed-sample denominators;
    unrecognized verdict handling and numerical clamping remain separate decisions.

### Known Issues and Pending Work

- **Candidate image integrity — user-approved correction (2026-10-25).** The old shared `Vis-<id>.jpeg` writers and existence-only reader sent unintended full-scene/stale inputs to the VLM; the uncorrected trace smoke run confirmed 15/19 input buffers were 640x800. Supported single-process VLM modes now use separate ordinary-frame paths, run/episode/search/candidate-specific composite paths, atomic publication, and explicit completion with identity/first-view/hash/format validation before API calls. Old files/archives remain untouched. See change entry 11 and the local README candidate-delivery section. New runs are image-corrected / Discriminator-corrected after change 12; retain the original labels of older archives.
- **Structured per-episode trace — user-approved implementation (2026-10-25).** Single-process `vlm_game` runs archive exact input-image buffers and per-episode JSONL events under a unique `logs/<experiment>/traces/<run-id>/` directory. The trace records prompts/configuration, logical sample returns/errors, actual aggregation pools and denominators, initial and checkpoint policies, decisions, and episode metrics. New runs identify candidate-local Discriminator aggregation; historical traces retain their actual legacy calculations and numerical anomalies. Entry 14 adds application-level API attempt diagnostics; SDK-internal retries, billed token usage, and hypothetical candidate-level navigation rewards are not recorded. See the local README trace-archive section and the change entries below.
- **API decoding/timeouts — historical failure and reliability correction.**
  The corrected GPT-4o-mini run completed 20/100 episodes, then stalled during
  episode 21/search 8 awaiting Discriminator sample `d-36-5`. Two earlier
  `JSONDecodeError` events (episode 7/search 5 Generator; episode 21/search 1
  Discriminator) came from decoding the outer HTTP response, not the model's
  `object_id` text. HTTP status/body diagnostics were not captured, so the exact
  upstream cause cannot be recovered. At that revision, `requests.post` had no
  timeout/status check, caught the wrong exception family, and malformed Generator
  content did not consume the retry budget. Entry 14 now adds connect/read timeouts,
  bounded application attempts, response checks, and diagnostics. It preserves
  the existing Requests/OpenAI SDK clients; it does not impose an absolute
  wall-clock deadline against continuously arriving bytes. The lingering process was
  stopped before verified archival under `saved_sessions/`; that directory also
  contains the old 38-episode VLM run and full CLIP baseline. See its README and
  the stopped mini session's `API-error-investigation.txt`. No API behavior was
  changed by that archival/investigation task; the subsequent reliability fix is
  separately documented below.
- **Baseline-first policy.** The current priority is to run the original CLIP-only language mode and obtain interpretable success/SPL/DTG results. Keep program logic frozen until the baseline is reviewed.

#### Discriminator response accumulation — approved algorithmic correction

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

## Local Setup

This section describes the current known-good local environment. Paths use
`$HOME` to avoid storing a username.

### Hardware and OS

- Ubuntu 22.04.5 LTS, x86_64
- Intel Core i7-11800H, 16 logical CPUs
- 38 GB RAM
- NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM
- NVIDIA driver 580.178.04
- X11 display available; GUI runs from the desktop session

The 6 GB GPU is tight. The full model stack uses about 5.8 GB while running.
Close other GPU-heavy applications before GUI or evaluation runs.

### Python Environment

- Miniforge at `$HOME/miniforge3`
- Conda env: `vlngame`
- Python 3.10.22
- `cmake` 3.25.3
- CUDA toolkit 12.1.66 (`nvcc`) inside the conda env
- CUDA comes from the conda environment

Activate with:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vlngame
export GSA_PATH="$HOME/Grounded-Segment-Anything"
```

`GSA_PATH` is required by `utils/model_utils.py` and
`utils/detection_segmentation.py`.

### Key Dependencies and Versions

- PyTorch 2.1.0+cu121
- torchvision 0.16.0+cu121
- torchaudio 2.1.0
- habitat-sim 0.2.1, headless build
- habitat-lab 0.2.1, editable install
- GroundingDINO, editable CUDA build from GSA
- segment-anything, editable install
- MobileSAM from GSA `EfficientSAM/MobileSAM` (symlinked as `MobileSAM`)
- open_clip_torch 2.24.0, CLIP ViT-H-14 `laion2b_s32b_b79k`
- Open3D 0.18.0
- numpy 1.23.5
- numpy-quaternion 2023.0.4
- transformers 4.36.2
- tokenizers 0.15.2
- huggingface-hub 0.25.2
- safetensors 0.4.5
- faiss-cpu 1.7.4
- supervision 0.21.0
- opencv-python 4.10.0.84; opencv-python-headless 4.11 is also present via supervision
- scikit-image 0.24.0, scikit-fmm 2023.4.2
- openai 0.28.1

`requirements.txt` remains the original repo file. The environment pins above
are installed in the conda env and recorded in
`/tmp/opencode/constraints.txt` for this machine.

### Model Weights and Data

Grounding-SAM assets:

- `$HOME/Grounded-Segment-Anything/groundingdino_swint_ogc.pth` (~662 MB)
- `$HOME/Grounded-Segment-Anything/EfficientSAM/mobile_sam.pt` (~39 MB)

The current MobileSAM setup uses only `mobile_sam.pt`.
CLIP ViT-H-14 weights are downloaded by `open_clip` on first use.

Datasets and scenes are local and untracked. HM3D scene assets live outside
the repo:

```text
$HOME/Documents/VLN-Game-hm3d/
  scene_data/                              # 100 HM3D val scenes + configs
  archives/                                # original HM3D tarballs
  metadata/                                # download metadata
```

They are exposed to the repo through:

```text
$HOME/VLN-Game/data/scene_datasets/hm3d -> $HOME/Documents/VLN-Game-hm3d/scene_data
```

The episode datasets remain under:

```text
$HOME/VLN-Game/data/datasets/
  objectgoal_hm3d/                         # 2,000 ObjectNav val episodes / 20 scenes
  vlobjectnav_hm3d_v4/                     # 316 language val episodes / 36 scenes
```

HM3D v0.1 val scene assets require a Matterport academic account/API token.
Do not record those credentials anywhere in the repo.

### Smoke Run Commands

Interactive launcher: `python demo.py` provides ObjectNav mode (reference), Language mode: CLIP-based, and Language mode: VLM. It always uses GUI mode, asks for episode count in all modes, and asks for a model only in VLM mode.

ObjectNav with GUI:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vlngame
export GSA_PATH="$HOME/Grounded-Segment-Anything"
cd "$HOME/VLN-Game"
python main_vis.py --episode_count 5 -v 1
```

CLIP-based language baseline (no API key or model):

```bash
python main_vln.py --vln_mode clip --episode_count 5 -v 0
```

Language mode: VLM requires an API key and the Open3D GUI. The local
provider and model configuration is stored in a private env file outside the
repo:

```bash
source "$HOME/.config/vlngame/api.env"
python main_vln.py -v 1
```

The code reads `VLN_API_KEY`, `VLN_API_BASE`, `VLN_MODEL`,
`VLN_MODEL_OPTIONS`, and `VLN_MODEL_SUPPORTS_TEMPERATURE` from the
environment. `demo.py` sets these from `VLN_MODELS`; each model entry names its
key variable. Do not pass keys through `--api` or store them in the repo.
The original OpenAI `--api` flow also works.

Do not use `main_vis_vec.py` or `main_vln_vec.py`; the vector variants
target an older API and remain outside the current single-process setup.

### Changes From the Original Paper Repo

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
     documented candidate-image filename collision remains pending.
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
      or multi-hour GUI stability. Artifacts: `/tmp/opencode/candidate-image-vlm-smoke/`.

12. **`utils/equ_ranking.py`, `utils/equ_trace.py`** — explicitly user-approved
    Discriminator aggregation correction affecting the equilibrium algorithm.
    - Reset each eligible candidate's `responses` and `response_ids` immediately
      before its queries. A=4/5 and B=2/5 now produce 0.8 and 0.4, rather than
      0.8 and 1.2. See the detailed three-part explanation above for cause,
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
      evidence of an improvement. Artifacts: `/tmp/opencode/discriminator-corrected-mini-smoke/`.

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

### Current Investigation

The immediate goal is to run `main_vln.py --vln_mode clip` on the original code and confirm the logged success/SPL/DTG results. Paper Table III reports the CLIP-based baseline as success 0.253, SPL 0.106, DTG 4.697. Program logic remains frozen until the local baseline is reviewed.

### Current Validation Status

ObjectNav GUI smoke run: 5 episodes, 4 successes, SPL 0.235. Episode 1
failed; episodes 2-5 succeeded. The Open3D GUI appeared and the manual query
box worked.

API smoke checks passed for the four configured models: GPT-4o, GPT-4o
mini, GPT-6 Luna, and DeepSeek V4.1 Flash. Each returned valid text JSON,
generator JSON, and discriminator plain text. GPT-4o required retries because
OpenRouter returned empty content on some attempts.

A 6-episode DeepSeek language batch reached the generator/equilibrium path in
all five new episodes; episode 5 accepted a candidate and succeeded (success
0.200, SPL 0.010). The batch predates the discriminator plain-text change and
the model registry, so a rerun is pending. Candidate-level behavior from that
batch is suspect because the candidate-image integrity issue remains
unresolved.

Language metrics reflect the selected provider; the paper used GPT-4o.

The GUI-close stop was tested on both entrypoints; both processes exited and
released the GPU. The interactive launcher now exposes the three demo modes;
ObjectNav launch, CLIP command construction without an API key, and VLM model
selection were tested. A local CLIP-only baseline run is still pending.
