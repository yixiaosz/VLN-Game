# Local setup

Last checked: **2026-11-01** (RTX 5090 host, fresh installation verified);
original RTX 3060 host recorded **2026-10-26**. This is a machine-specific
reproduction guide, not a portable dependency lock. See [AGENTS.md](../AGENTS.md)
for editing rules and [reproduction status](REPRODUCTION_STATUS.md) for what
actually completed.

## Hardware and OS

Two hosts have run the full pipeline. The **RTX 5090 host is the latest running
setup**; the original RTX 3060 laptop setup also works and produced every archive
before 2026-10-31.

| | RTX 5090 host (current) | RTX 3060 laptop (original, also works) |
|---|---|---|
| OS / kernel | Ubuntu 24.04.5 LTS, 7.0.0-42 | Ubuntu 22.04.5 LTS |
| CPU / RAM | AMD Ryzen 9 9950X3D, 32 logical CPUs; 60 GB | Intel Core i7-11800H, 16 logical CPUs; 38 GB |
| GPU / driver | GeForce RTX 5090 (Blackwell, sm_120), 32,607 MiB; 595.99.02 (`nvidia-driver-595-open`) | GeForce RTX 3060 Laptop, 6,144 MiB; 580.178.04 |
| System compiler | gcc 13.3 | (not recorded) |
| Verified by | 98/98 offline tests, CLIP 5-episode smoke, full 316-episode 1017 Game run (2026-10-31) | All runs up to the 2026-10-30 cohort |

Both use X11 for GUI runs; headless VLM runs do not require DISPLAY. On the
6 GB laptop GPU the full model stack uses roughly 5.8 GB, so close other
GPU-heavy applications first; the RTX 5090 run peaked around 6.5 GB of 32 GB.

## Python environment

- Miniforge: `$HOME/miniforge3`; conda environment: `vlngame`.
- Python 3.10.22; CMake 3.25.3 on both hosts.
- CUDA toolkit inside the conda environment: 12.8 (`nvidia/label/cuda-12.8.1`)
  on the RTX 5090 host; 12.1.66 on the RTX 3060 host.

From the repository root:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vlngame
export GSA_PATH="$HOME/Grounded-Segment-Anything"
```

`GSA_PATH` is required by `utils/model_utils.py` and
`utils/detection_segmentation.py`.

### Recorded dependencies

| Component | RTX 5090 host (current) | RTX 3060 host (original) |
|---|---|---|
| PyTorch / torchvision / torchaudio | 2.7.1+cu128 / 0.22.1+cu128 / 2.7.1+cu128 | 2.1.0+cu121 / 0.16.0+cu121 / 2.1.0 |
| habitat-sim | 0.2.1, headless build with system gcc 13 | 0.2.1, headless build |
| habitat-lab | 0.2.1, editable install | 0.2.1, editable install |
| GroundingDINO | Editable CUDA build for sm_120, `scalar_type()` fix | Editable CUDA build from GSA |
| segment-anything | Editable install | Editable install |
| pytorch3d | 0.7.8 (`stable`), CUDA build for sm_120 | (not recorded) |
| MobileSAM | GSA `EfficientSAM/MobileSAM`, loaded via `sys.path` | GSA `EfficientSAM/MobileSAM`, symlinked as `MobileSAM` |
| open_clip_torch | 2.24.0; ViT-H-14 `laion2b_s32b_b79k` | same |
| Open3D | 0.18.0 (EGL headless confirmed) | 0.18.0 |
| numpy / numpy-quaternion | 1.23.5 / 2023.0.4 | same |
| transformers / tokenizers | 4.36.2 / 0.15.2 | same |
| huggingface-hub / safetensors | 0.25.2 / 0.4.5 | same |
| faiss-cpu / supervision | 1.7.4 / 0.21.0 | same |
| opencv-python | 4.10.0.84 | 4.10.0.84; headless 4.11 also present via supervision |
| scikit-image / scikit-fmm | 0.24.0 / 2023.4.2 | same |
| openai | 0.28.1 | same |

### RTX 5090 host build notes (2026-10-31)

Blackwell (sm_120) needs PyTorch 2.7+ with CUDA 12.8; the 2.1/12.1 pins cannot
run on it. The CUDA toolkit's conda gcc cannot see the system OpenGL/EGL
libraries, so habitat-sim is built with the system compiler and gcc 13 needs
standard headers its 2021 bundled dependencies omit. Sources stay outside the
repository (`~/habitat-sim`, `~/habitat-lab`, `~/Grounded-Segment-Anything`).
Privileged prerequisites (run by the user): `git git-lfs build-essential cmake
libegl1-mesa-dev libgl1-mesa-dev libglfw3-dev` and `nvidia-driver-595-open`.

```bash
# habitat-sim, from ~/habitat-sim at tag v0.2.1 (setuptools<70 in the env)
env -u CMAKE_ARGS -u CFLAGS -u CPPFLAGS -u LDFLAGS -u AR -u RANLIB -u LD \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  CXXFLAGS="-include cstdint -include tuple -include limits -include stdexcept -include string -include utility -include algorithm" \
  python setup.py install --headless
# CUDA extensions (GroundingDINO, pytorch3d)
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1
pip install --no-build-isolation -e "$HOME/Grounded-Segment-Anything/GroundingDINO"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

GroundingDINO's `ms_deform_attn_cuda.cu` needs `value.type()` →
`value.scalar_type()` (two dispatch lines) for PyTorch 2.7; upstream made the
same fix. The HM3D scenes and episode datasets were copied from the original
host, keeping its documented data compatibility edits.

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

**DeepSeek V4.1 Flash 1017 via OpenRouter (RTX 5090 host, 2026-10-31).** Direct
`api.deepseek.com` is unreachable from that host's network (TLS reset for all
deepseek.com hosts), so its full Game run used OpenRouter, pinned to the DeepSeek
provider. That provider requires the account's OpenRouter privacy setting to
allow paid-training providers. Effective settings (no registry entry):

```bash
export VLN_MODEL="deepseek/deepseek-v4.1-flash-1017"
export VLN_API_BASE="https://openrouter.ai/api/v1"
export VLN_API_KEY="$OPENROUTER_API_KEY"
export VLN_MODEL_OPTIONS='{"reasoning":{"enabled":false},"provider":{"only":["deepseek"],"allow_fallbacks":false}}'
export VLN_MODEL_SUPPORTS_TEMPERATURE=1
```

Set `VLN_MODEL_OPTIONS` explicitly: otherwise a `deepseek...` model name gets the
direct-API `thinking` option, which is not OpenRouter's parameter. Keep this route
labelled separately from direct `deepseek-flash` results.

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
Raw v1 generation responses/provenance are retained with the completed Jev Game
run in `configuration/parse-generation/` (relocated from the retired smoke tree).

To prepare a **new version**, load `DEEPSEEK_API_KEY` privately, then run:

```bash
python -m utils.prepare_jev_lookup --archive saved_sessions/jev-game-next/configuration/parse-generation \
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

## Archive retention and post-run cleanup

Approved **2026-10-30**, change 22. Keep completed full benchmarks (316 language
or 2,000 ObjectNav episodes) and the longest GPT-4o-mini attempt; the current
exception is 188/316, including its episode-189 failure evidence. Prune standalone
short/failed/smoke sessions after relevant retained evidence is consolidated.

Keep each retained run's exact candidate JPEGs (trace inputs and published
copies), raw responses, JSONL/text reports, source/configuration snapshots and
its own diagnostics/monitoring. Preserve frozen-parser generation evidence inside
the corresponding full Jev archive; `configs/` remains the runtime table location.
For final archives, remove `output/dump/` similarity-map PNG trees and equivalent
heatmap `.tar.gz` files/member manifests. These are lossy per-step diagnostic
exports, not VLM candidate composites. This applies to future runs too; it does
not authorize changing navigation or runtime map calculations.

Retain byte-exact run-only evaluation logs; remove foreign appended launch blocks
or mixed copies. Consolidate duplicate root monitoring into the owning sessions.
Use reviewed allowlists and preserve all as-measured source/documentation
snapshots. Update current metadata/README/indexes, but retain original acquisition
counts, retired inventory digests and cleanup accounting. A new seal must be
labelled **post-cleanup**, not represented as the original acquisition seal.

The current cohort's local `saved_sessions/cleanup_sessions.py` records the scope
in `cleanup-plan.json`, preserves byte fingerprints, applies cleanup and finally
reseals once. It deliberately uses an explicit cohort allowlist; review/extend
the scope and retain dated records for future cohorts rather than blindly
rerunning this one-time cleanup. Read-only inspection after resealing:

```bash
python saved_sessions/cleanup_sessions.py verify
```

This verifier is scoped to the eight 2026-10-30 cohort archives and asserts that
exact folder set. Since the 2026-10-31 session was added, it stops at that assertion
(change 24). Verify newer sessions with their own read-only script:

```bash
python saved_sessions/20261031-deepseek-v4.1-flash-1017-openrouter-316ep/verify_session.py
```

Do not rerun old per-archive `verify_final.py`, `seal_archive.py` or launch writers:
they describe the original acquisition and may overwrite historical map counts,
fail old inventory-count assertions or modify sealed data. Use the cleanup-aware
read-only verifier. Current counts/digests are in `saved_sessions/cleanup-record.json`
and each session's `archive-seal.json`; original metric values remain unchanged.

### Renamed archive folders

The user shortened the eight outer session-directory names on 2026-10-30,
including removal of the redundant headless tag. Current locations are indexed
in `saved_sessions/catalog.json` and README. `saved_sessions/session-renames.json`
maps original acquisition/cleanup names to current directories; the read-only
verifier resolves those aliases without changing historical records.

Raw JSONL, launch/source/documentation snapshots, reports and dated cleanup
records retain their original embedded paths as provenance. Use the alias map
when locating a recorded old absolute path. Do not globally replace paths inside
sealed files. Outer-folder-only renames leave relative SHA256 inventories valid;
all archive bytes, metrics and existing seals were verified unchanged, so no
resealing was required. This differs from deleting/editing archive contents.

## Maintenance

Update this document only for verified setup, asset-path or supported-invocation
changes. Date the observation and state whether it came from an actual check or
a historical record. Keep compatibility rationale in the change history and
experiment outcomes in the status page/session archives. Follow the
[documentation maintenance rules](../AGENTS.md#documentation-maintenance).
