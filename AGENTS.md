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

### Known Issues and Pending Work

- **Candidate image integrity — pending confirmation.** The Open3D GUI has two writers for the same `Vis-<id>.jpeg` filename pattern: `on_top_rendered()` writes a full-scene 640×800 image using the frame index, while `on_image_rendered()` writes the intended 960×480 top-view plus first-view composite using the candidate ID. The agent reads `Vis-<id>.jpeg` by file existence only. Most candidate files in the last run were 640×800. Treat VLM candidate-image inputs as suspect until confirmed. Do not fix this issue unless the user explicitly asks.
- **Structured per-episode trace — pending feature.** The full VLM inputs, prompts, generator/discriminator raw responses, policy arrays, probability arrays, selected candidates, and candidate-level rewards are not persisted in a structured trace. Do not implement this feature unless the user explicitly asks.
- **Baseline-first policy.** The current priority is to run the original CLIP-only language mode and obtain interpretable success/SPL/DTG results. Keep program logic frozen until the baseline is reviewed.

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
