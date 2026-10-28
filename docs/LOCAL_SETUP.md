# Local setup

Last checked against the recorded local setup: **2026-10-26**. This is a
machine-specific reproduction guide, not a portable dependency lock or a fresh
installation verification. See [AGENTS.md](../AGENTS.md) for editing rules and
[reproduction status](REPRODUCTION_STATUS.md) for what actually completed.

## Hardware and OS

- Ubuntu 22.04.5 LTS, x86_64.
- Intel Core i7-11800H, 16 logical CPUs; 38 GB RAM.
- NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM.
- NVIDIA driver 580.178.04.
- X11 desktop available for GUI runs; headless VLM runs do not require DISPLAY.

The 6 GB GPU is tight: the full model stack uses roughly 5.8 GB while running.
Close other GPU-heavy applications before navigation runs.

## Python environment

- Miniforge: `$HOME/miniforge3`; conda environment: `vlngame`.
- Python 3.10.22; CMake 3.25.3.
- CUDA toolkit 12.1.66 (`nvcc`) is installed inside the conda environment.

From the repository root:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vlngame
export GSA_PATH="$HOME/Grounded-Segment-Anything"
```

`GSA_PATH` is required by `utils/model_utils.py` and
`utils/detection_segmentation.py`.

### Recorded dependencies

| Component | Recorded version/setup |
|---|---|
| PyTorch / torchvision / torchaudio | 2.1.0+cu121 / 0.16.0+cu121 / 2.1.0 |
| habitat-sim | 0.2.1, headless build |
| habitat-lab | 0.2.1, editable install |
| GroundingDINO | Editable CUDA build from GSA |
| segment-anything | Editable install |
| MobileSAM | GSA `EfficientSAM/MobileSAM`, symlinked as `MobileSAM` |
| open_clip_torch | 2.24.0; ViT-H-14 `laion2b_s32b_b79k` |
| Open3D | 0.18.0 |
| numpy / numpy-quaternion | 1.23.5 / 2023.0.4 |
| transformers / tokenizers | 4.36.2 / 0.15.2 |
| huggingface-hub / safetensors | 0.25.2 / 0.4.5 |
| faiss-cpu / supervision | 1.7.4 / 0.21.0 |
| opencv-python | 4.10.0.84; headless 4.11 also present via supervision |
| scikit-image / scikit-fmm | 0.24.0 / 2023.4.2 |
| openai | 0.28.1 |

`requirements.txt` remains the original paper-repo file; it is not a lock for the
local environment. Reasons for the local pins and build adjustments are retained
under [environment-only changes](CHANGES_FROM_ORIGINAL.md#environment-only-changes).
A former setup note referenced `/tmp/opencode/constraints.txt`; that file was
absent when this documentation was reorganized. Do not rely on it as an install
artifact. Session archives may contain explicitly dated package-version captures;
those are evidence of that environment, not complete replacement lockfiles.

## Weights and data

Grounding-SAM assets:

- `$HOME/Grounded-Segment-Anything/groundingdino_swint_ogc.pth` (~662 MB).
- `$HOME/Grounded-Segment-Anything/EfficientSAM/mobile_sam.pt` (~39 MB).

Only `mobile_sam.pt` is used for MobileSAM. `open_clip` downloads CLIP ViT-H-14
weights on first use if they are not already cached.

HM3D assets are local, untracked, and outside the repository:

```text
$HOME/Documents/VLN-Game-hm3d/
  scene_data/       # 100 HM3D val scenes + configs
  archives/         # original HM3D tarballs
  metadata/         # download metadata
```

The repository exposes them through this symlink:

```text
data/scene_datasets/hm3d -> $HOME/Documents/VLN-Game-hm3d/scene_data
```

Episode datasets remain inside the repository's untracked data directory:

```text
data/datasets/objectgoal_hm3d/      # 2,000 ObjectNav val episodes / 20 scenes
data/datasets/vlobjectnav_hm3d_v4/  # 316 language val episodes / 36 scenes
```

HM3D v0.1 val assets require Matterport academic-account access. Never record its
credentials in the repository. The compatibility edits to downloaded episode
JSON and directory names are documented under
[data compatibility changes](CHANGES_FROM_ORIGINAL.md#data-compatibility-changes).
Do not repeat transformations blindly or alter goals, poses, instructions or metrics.

## Supported entrypoints and smoke runs

Use single-process `main_vis.py` / `main_vln.py`. The vector variants
`main_vis_vec.py` and `main_vln_vec.py` target an older API and are outside the
supported local setup.

```bash
# ObjectNav reference with GUI
python main_vis.py --episode_count 5 -v 1

# CLIP-only language baseline; no API key or model needed
python main_vln.py --vln_mode clip --episode_count 5 -v 0

# VLM: load and select the private API configuration first (see below)
python main_vln.py --vln_mode vlm_game --episode_count 5 -v 0

# Interactive VLM GUI
python main_vln.py --vln_mode vlm_game --episode_count 5 -v 1

# Generator (one sample) / Ranking (five-sample plurality), same API environment
python main_vln.py --vln_mode vlm --episode_count 5 -v 0
python main_vln.py --vln_mode vlm_rank --episode_count 5 -v 0
```

All three VLM methods use the same headless/GUI image delivery and navigation.
Generator/Ranking trace their actual sample counts/decisions, with Discriminator
and equilibrium marked not applicable. Ranking ties follow input candidate order
with reject (`-1`) last. Their baseline-only ID validation regenerates unknown or
non-string IDs within the existing shared three-attempt budget, aborting only on
exhaustion, without changing the Game decoder. See changes 18 and 19 in the history.
The launcher offers all three VLM methods, with Game as the default; the CLI flags also remain available.

`python demo.py` first asks for headless or GUI operation, then ObjectNav, CLIP
language or VLM language mode. The display choice applies to all three modes:
headless passes `-v 0` without requiring DISPLAY, while GUI passes `-v 1` and checks
for desktop access. It asks for episode count in every mode. VLM mode then asks
for VLN-generator / VLN-ranking / VLN-game (Enter defaults to Game; `b` returns to
episode count), followed by the model. This applies to Jev and existing chat
models. The summary confirms display/method/model and the final log path follows
the selected method. GUI-close stops a GUI run;
manual query and pause/resume controls are preserved.

### Private API configuration

The private provider/model environment is outside the repository:

```bash
source "$HOME/.config/vlngame/api.env"
```

Supported calls read `VLN_API_KEY`, `VLN_API_BASE`, `VLN_MODEL`,
`VLN_MODEL_OPTIONS` and `VLN_MODEL_SUPPORTS_TEMPERATURE`. Configure these for the
intended provider/model before a direct CLI run. Sourcing a `VLN_MODELS` registry
alone does not select a model: `demo.py` translates its chosen registry entry into
the effective `VLN_*` values. Each registry entry names its key variable.
Do not print key values or pass them in `--api`/argv. The original OpenAI `--api`
flow is supported by the code but is not the recommended secret-handling method.

Historical supported model configurations include OpenRouter GPT-4o, GPT-4o mini,
GPT-6 Luna and direct DeepSeek `deepseek-flash`. Luna omits temperature and uses
`reasoning.effort=none`; DeepSeek uses `thinking.type=disabled`. See change 4 in
[the change history](CHANGES_FROM_ORIGINAL.md) for provenance. Past API smoke
checks do not certify current provider availability. The paper used GPT-4o;
results from other models/providers must retain their actual model labels.

### Jev 1.14 backend and frozen parsing

Select **Jev 1.14** in `demo.py`, or configure the effective environment for direct
CLI use after sourcing the private API environment:

```bash
export VLN_MODEL="typesafe/jev-1.14"
export VLN_API_KEY="$OPENROUTER_API_KEY"
export VLN_API_BASE="https://openrouter.ai/api/v1"
export VLN_MODEL_OPTIONS='{}'
export VLN_MODEL_SUPPORTS_TEMPERATURE=0
python main_vln.py --vln_mode vlm_game --episode_count 3 -v 0
# Other methods retain their names/flags: vlm (VLN-generator), vlm_rank (VLN-ranking).
```

Jev uses `https://openrouter.ai/api/alpha/decisions`, not chat completions or the
Jev Router. The native adapter omits temperature and other chat-generation
options. Existing registry entries are preserved; Jev is appended if a private
`VLN_MODELS` registry does not already contain it. Selecting another registry
model resets its effective key/base/options/temperature settings as before.

Only Jev uses `configs/jev_instruction_parses.v1.json` and its `.json.sha256`
sidecar. Override with `VLN_JEV_PARSE_TABLE` only for another explicitly prepared
version. Preflight checks the checksum, exact instruction coverage and dataset
fingerprints. There is no live parsing fallback. Other models still parse live.
The 316-entry v1 table was generated by direct DeepSeek Flash using the unchanged
parser prompt/settings (temperature 0.5, 512 tokens, disabled thinking). Structural
validity is not a semantic accuracy guarantee. Parsed phrases feed existing
perception/target similarity; Jev receives the original instruction and images.

To prepare a **new version**, load `DEEPSEEK_API_KEY` privately, then run:

```bash
python -m utils.prepare_jev_lookup --archive saved_sessions/jev-parses-v2 \
  --output configs/jev_instruction_parses.v2.json
```

The tool preserves raw provider responses and completed-entry journals, resumes
matching unfinished generation, and refuses to overwrite a published table or
checksum. New dataset instructions require a new fully covered table version.
Jev methods, solver semantics and validation evidence are in change 20 of the
[change history](CHANGES_FROM_ORIGINAL.md); current smoke outcomes are in the
[status page](REPRODUCTION_STATUS.md).

### Rendering and API limits

- `-v 0` VLM uses one persistent synchronous Open3D offscreen renderer; no DISPLAY
  is needed on this machine. `-v 1` uses the GUI. Both still use Filament.
- Offscreen uses a fixed 640x800 GUI-reference viewport and the shared 960x480
  candidate composite. Small cross-backend pixel differences remain.
- GUI candidate completion waits are bounded by 120 seconds. A synchronous
  offscreen native render has no enforceable Python deadline or crash isolation.
- Supported API calls use 10/90-second connect/read-inactivity timeouts and at
  most three application attempts, 2/4-second backoff and Retry-After capped at
  30 seconds. These are not absolute wall-clock deadlines; SDK-internal retries
  and billed token counts are not individually traced.
- Required API-sample exhaustion aborts without partial-batch aggregation.
  Rendering/publication errors also abort explicitly; incomplete episodes have
  null metrics and are excluded. Instruction parsing additionally requires a
  string target and an optional list of string landmarks; invalid shapes use the
  same bounded retry budget (change 16). This is type checking, not semantic
  repair or a guarantee that a structurally valid description is correct.

## Maintenance

Update this document only for verified setup, asset-path or supported-invocation
changes. Date the observation and state whether it came from an actual check or
a historical record. Keep compatibility rationale in the change history and
experiment outcomes in the status page/session archives. Follow the
[documentation maintenance rules](../AGENTS.md#documentation-maintenance).
