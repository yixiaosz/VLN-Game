# VLN-GAME Reproduction and Probing

This fork reproduces the VLN-Game algorithm for local experimentation on Habitat/HM3D. The ObjectNav pipeline is the reference check for the environment, perception, mapping, and GUI. The language-goal VLM game is the main focus.

See [AGENTS.md](AGENTS.md) for agent rules and document maintenance,
[local setup](docs/LOCAL_SETUP.md) for dependencies/data/API configuration,
[change history](docs/CHANGES_FROM_ORIGINAL.md) for approved corrections, and
[reproduction status](docs/REPRODUCTION_STATUS.md) for current blockers and results.

## Before running

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vlngame
export GSA_PATH="$HOME/Grounded-Segment-Anything"
cd "$HOME/VLN-Game"
```

For Language mode: VLM, load the private API environment:


```bash
# just an example
source "$HOME/.config/vlngame/api.env"
```

## Quick launcher

```bash
python demo.py
```

The launcher's first question selects **Headless (no window)** or **GUI (Open3D)**.
It then asks for ObjectNav (reference), CLIP-based language, or VLM language mode,
followed by the 5/full/custom episode count and a model only in VLM mode. The
display choice applies to every mode and is shown again in the launch summary.
Headless runs do not require `DISPLAY`; GUI runs require desktop access.

## ObjectNav demo

`main_vis.py` is the single-process ObjectNav entrypoint. Add `-v 1` to open the Open3D window; `-v 0` runs headless.

| Run | Command |
|---|---|
| 5 episodes (episodes 1-5) | `python main_vis.py --episode_count 5 -v 1` |
| Custom count (example: 10 episodes from #1) | `python main_vis.py --episode_count 10 -v 1` |
| Full run (2,000 episodes) | `python main_vis.py -v 1` |

Close the Open3D window to stop the run. A full ObjectNav run takes days on the local GPU; use the short demos first.

## Language modes (main focus)

For Language mode: CLIP-based, run `python main_vln.py --vln_mode clip --episode_count 5 -v 0`. This mode requires no API key and no model.

`main_vln.py` parses the instruction, builds the object-centric map, proposes CLIP candidates, and runs the generator/discriminator equilibrium search. The default mode is `vlm_game`, which is the paper's game-theoretic mode. Other modes are available through `--vln_mode`.

| VLM method | Flag | Identification per search |
|---|---|---|
| VLN-Generator | `--vln_mode vlm` | One Generator sample; select its actual object ID or reject (`-1`) |
| VLN-Ranking | `--vln_mode vlm_rank` | Five Generator samples; highest vote count wins |
| VLN-Game (default) | `--vln_mode vlm_game` | Existing Generator/Discriminator equilibrium search |

Generator and Ranking share the same images, prompt and navigation pipeline.
Ranking uses plurality, not a requirement for three votes: ties use the original
candidate order, with `-1` last (so a tied real candidate wins over rejection).
Malformed/unknown/non-string IDs abort baseline selection rather than silently
turning into rejection votes; this baseline-only check does not change Game's
current interpretation. Transport/model-syntax retries retain their existing
budget; invalid candidate IDs do not introduce an extra retry loop.

All three VLM methods archive exact image inputs, raw opinions, API attempts,
decisions and episode metrics. Generator/Ranking record zero Discriminator calls
and zero equilibrium updates, labelled not applicable. Use direct commands to
choose the baseline; `demo.py`'s VLM option continues to select Game by default.

```bash
python main_vln.py --vln_mode vlm --episode_count 5 -v 0
python main_vln.py --vln_mode vlm_rank --episode_count 5 -v 0
```

VLM mode supports both the Open3D GUI (`-v 1`) and candidate-only offscreen
rendering (`-v 0`). Headless mode requires no `DISPLAY` or desktop window.

| Run | Command |
|---|---|
| 5 episodes (episodes 1-5) | `python main_vln.py --episode_count 5 -v 1` |
| 5 episodes, headless | `python main_vln.py --vln_mode vlm_game --episode_count 5 -v 0` |
| Custom count (example: 10 episodes from #1) | `python main_vln.py --episode_count 10 -v 1` |
| Full run (316 episodes) | `python main_vln.py -v 1` |

Practical notes:

- Closing the Open3D window stops the run.
- The full VLM run covers 316 episodes and has API cost. Start with 5 episodes.

### Headless VLM rendering

With `-v 0`, a single persistent Open3D `OffscreenRenderer` renders candidate
views synchronously on the navigation thread, only when a VLM search needs them.
There is no interactive window, GUI-update thread, or per-frame GUI screenshot
rendering. Habitat, mapping, candidate selection, first-view selection, API
settings, equilibrium equations, and evaluation semantics are preserved.

The offscreen reference is the untouched local GUI's 640x800 scene viewport,
90-degree vertical field of view, camera offset, clipping bounds, materials,
white background, and red candidate bounding box. Both paths share the original
960x480 composite operation. The headless viewport is fixed rather than dependent
on desktop scaling or window resizing. Manifests record the renderer backend,
reference and camera settings; historical runs keep their original labels.

The installed Open3D 0.18.0 EGL backend works without `DISPLAY`; no dependency
replacement or software-rendering environment override is needed on this machine.
`demo.py` supports both headless and GUI runs; select the display mode at its
first question. Direct commands remain available. Load the same private API
environment before VLM runs in either display mode.

Offscreen rendering still uses Filament. Removing GUI callbacks and per-frame
rendering reduces the workload but does not guarantee absence of native crashes.
A synchronous native render has no enforceable Python timeout; the GUI's
120-second completion wait does not apply. Python rendering/publication errors
stop the test explicitly with an incomplete episode; native crashes can still
terminate the process. Completed episode traces are flushed to disk.

Initial validation passed all 68 offline tests and a real three-episode GPT-4o-mini
headless smoke run in 6m40s. Its 11 API input images and policy checkpoints passed
an independent audit. GUI/offscreen camera matrices matched exactly in six
reference renders; small native pixel differences remain, comparable in mean to
GUI repeat-render variation. The four-hour renderer-only test was cancelled at
the user's request and replaced with a 100-episode GPT-4o-mini headless run;
multi-hour stability is not yet established. See the
[reproduction status](docs/REPRODUCTION_STATUS.md) for outcomes and evidence,
including the later 70/100-episode headless run stopped by a parsing-structure error.

### Candidate image delivery

Single-process VLM runs use request-specific candidate images rather than
polling shared `Vis-<id>.jpeg` files. Ordinary GUI screenshots are written under
`<dump>/<experiment>/episodes/<episode>/frames/Frame-<frame>.jpeg`; they can have
different dimensions and are never used as candidate inputs.

Candidate composites are published under:

```text
<dump>/<experiment>/episodes/<episode>/candidates/<run-id>/search-<search>/candidate-<id>.jpeg
```

Navigation waits for an explicit completion matching the run, episode, search,
candidate, and selected first-view hash. A fully written JPEG is published
atomically, then read once and checked against the completion hash and expected
960x480 dimensions before encoding for the API. The existing two-panel layout,
camera/bounding-box settings, and selection of the object's first-view image are
preserved. Old `Vis-*.jpeg` files are ignored, not deleted or resized.

Render errors stop the affected test with an error instead of substituting an
image or manufacturing a navigation result. GUI closure cancels pending waits;
GUI render completion has a 120-second per-image timeout. Offscreen calls are
synchronous without an enforceable native deadline. Traces record the protocol
`request_scoped_atomic_completion_v1` and each image's completion identity.
New runs are image-corrected / Discriminator-corrected. Preserve the original
labels of older image-corrected / Discriminator-uncorrected runs.

### Discriminator aggregation correction (algorithmic)

1. **Bug:** the original response list was shared across candidates. If A received
   four `correct` responses out of five and B received two, B was scored as
   `(4 + 2) / 5 = 1.2` instead of `2 / 5 = 0.4`.
2. **Why fix before reproduction:** a frequency can exceed 1.0. With the 0.05
   bias and normalization, the example creates a negative policy entry; the
   subsequent logarithm can produce NaNs. Even in-range contamination can change
   equilibrium choices and navigation trajectories, invalidating evaluation of
   the intended algorithm. This inherited source bug does not establish which
   revision produced the paper's metrics, and correction does not guarantee a
   higher score or equality with the paper's results.
3. **Fix:** reset both the response list and its trace sample IDs for each
   eligible candidate immediately before its five queries. Each score now uses
   only that candidate's returned batch. Prompts, sampling, bias, equilibrium
   equations, reject-all branches, fixed-five Discriminator denominator, and
   invalid-response handling are otherwise preserved. No clamping is added.

The user explicitly approved this correction after lifting the earlier hold.
New manifests identify it as `candidate_local_responses_v1`; keep historical
uncorrected archives separately labelled. See the
[change history](docs/CHANGES_FROM_ORIGINAL.md) for detailed provenance and validation.

The GUI annotation helper also uses the supported Supervision bounding-box and
label APIs, preserving the previous pixels/styles while eliminating the repeated
`BoxAnnotator.annotate` deprecation warning. This is a separate visualization
compatibility change; Supervision remains pinned to the installed 0.21.0 setup.

### API reliability

Supported parsing/Generator/Discriminator calls keep the existing OpenAI SDK and
Requests clients, with explicit 10-second connection and 90-second read-inactivity
timeouts. A shared, finite loop allows at most three application attempts, with
2/4-second retry delays and Retry-After capped at 30 seconds. Transient HTTP,
connection, decoding, model-syntax and instruction-parsing structure failures consume this
budget; permanent request/authentication/configuration errors fail immediately.

HTTP status and response structure are checked before content extraction. Trace
events `api_attempt_start` / `api_attempt_end` link retries to logical sample IDs
and retain status, Content-Type, request ID, elapsed time, failure category, and
bounded sanitized failure previews. Authorization headers/keys and request-image
base64 are not logged. Terminal reports include retry/failure diagnostics.

If a required sample ultimately fails, the test stops with an explicit API error:
the affected episode is incomplete, not counted as a navigation failure, and no
partial-batch frequency is calculated. Previously completed episodes remain saved.
This is labelled `abort_test_no_partial_batch` in the manifest; old archives keep
their actual Generator/Discriminator denominator behavior. Complete batches still
use five logical samples and the same prompts, payload settings, and equations.

These are connection/read timeouts, **not a strict wall-clock deadline** against
a server continuously sending bytes. Interrupts stop peer retry loops before
executor shutdown, but cannot forcibly cancel an active HTTP thread. SDK-internal
connection retries are not individually traced. This is intentionally a focused
reliability patch rather than a new asynchronous transport/cancellation framework.

Instruction parsing also checks that the decoded value is a dictionary with a
string `target` and, when present, a list containing only string `landmark` entries.
Invalid shapes are labelled `parsing_structure` and regenerate the same request
within the existing three-attempt budget. Values are never flattened, dropped or
coerced; missing/empty landmark lists and valid response text retain their behavior.
Manifests identify this as `target_string_optional_landmark_string_list_v1`.
See [change 16](docs/CHANGES_FROM_ORIGINAL.md) and the
[current verification status](docs/REPRODUCTION_STATUS.md).
The exact archived episode-71 malformed response was rejected and recovered via
a live retry; a separate natural-provider replay of that dataset episode then
completed successfully. This targeted check does not complete the stopped
100-episode benchmark.

### VLM-Game trace archive

Single-process `main_vln.py` runs in `vlm`, `vlm_rank` or `vlm_game` automatically create a
unique trace directory below `dump/logs/vlobjectnav-vlm_game/traces/` (or the
corresponding `--dump_location` and experiment name). `eval.log` prints the
run directory and short references for each episode/search.

```text
traces/<run-id>/
  run.json
  episodes/001/
    equilibrium.jsonl
    equilibrium.txt
    images/search-001-input-000-candidate-15.jpeg
```

`run.json` records effective model/request settings, exact prompts, task config,
and source hashes without API credentials. Each episode's JSONL records parsing,
search inputs, individual returned samples/errors, aggregation pools and
denominators, initial policies, updates 125/250/375/500, selected candidates,
and final episode metrics. Raw response text and its parsed interpretation are
separate events linked by sample ID. Records are flushed as they are written;
an absent `search_end` or `episode_end` indicates an incomplete operation.

Archive **the entire run directory alongside `eval.log`**, rather than just the
mutable GUI image directory. Each archived image is the exact byte buffer
encoded for the VLM, with its hash, dimensions, source path, and input position.
Historical archives can contain unexpected-sized or unreadable buffers; these
remain preserved and flagged, not substituted or rewritten. New candidate
requests use the validated completion protocol above, so invalid candidate
images are rejected before an API call. This corrects the previous shared-file
overwrite/stale-read issue in the supported single-process VLM path.

Discriminator batch-local sample IDs and actual scoring pools are recorded
separately; new candidate-local pools match their candidate's returned samples.
Historical traces retain their accumulated pools and numerical anomalies.
Nonfinite numerical values are represented by the JSON strings
`"NaN"`, `"Infinity"`, and `"-Infinity"`, with policy diagnostics; no values are
clamped or repaired. Application-level API attempts are now traced separately
from logical samples; SDK-internal retries and billed token usage are not captured. CLIP,
Generator-only, and Ranking modes retain their existing logging.

#### Terminal-readable episode reports

After an episode trace closes, `equilibrium.txt` is generated alongside its
JSONL. It contains an episode/search overview, raw opinions with sample IDs,
actual aggregation pools and denominators, image paths/hashes, initialization
inputs, and complete Generator/Discriminator policy tables at every recorded
checkpoint. Unexpected image sizes, reused response pools, nonfinite/negative
policies, query errors, and incomplete operations are highlighted.

Reports are plain ASCII, wrapped to 100 columns. Displayed numeric fields use
10 significant digits and scientific notation where appropriate; JSONL remains
the exact, full-precision record. Text conversion does not recompute probabilities,
call the VLM, or modify JSONL/images. A report-generation failure is logged without
changing episode outcomes.

Generate or regenerate reports for an existing run or a single episode:

```bash
python -m utils.equ_report "/path/to/run-directory"
python -m utils.equ_report "/path/to/episodes/001/equilibrium.jsonl"
less -S "/path/to/episodes/001/equilibrium.txt"
```

The converter also accepts a directory containing multiple runs. It marks
missing completion events and warns about a truncated, unterminated final JSONL
record. Invalid records earlier in a file cause an error rather than being
silently skipped. Only derived `.txt` files are written or replaced.

Offline trace/parity checks (no API calls):
`python -m unittest discover -s tests -v`.

---

(Below this line is the original README of the [VLN-game](https://github.com/ybgdgh/VLN-Game) code)

---

# VLN-Game: Vision-Language Equilibrium Search for Zero-Shot Semantic Navigation

[**Project Page**](https://sites.google.com/view/vln-game) |[**ArXiv**](https://arxiv.org/pdf/2411.11609) |[**Video**](https://youtu.be/YgkyYVy6tX8)

We proposed a new framework to explore and search for the language descriptive targets in unknown environment based on Large Vision Language Model. Our work is based on [Concept-Graph](https://github.com/concept-graphs/concept-graphs?tab=readme-ov-file) and [L3MVN](https://sites.google.com/view/l3mvn), implemented in PyTorch. Part of this work was conducted during the first author’s internship at Tencent Robotics X.

**Author:** Bangguo Yu<sup>1, 2</sup>, Yuzhen Liu<sup>1</sup><sup>†</sup>, Lei Han<sup>1</sup>, Hamidreza Kasaei<sup>2</sup><sup>†</sup>, Tingguang Li<sup>1</sup><sup>†</sup>, and Ming Cao<sup>2</sup>

**Affiliation:** <sup>1</sup>Tencent Robotics X     <sup>2</sup>University of Groningen


## Abstract

Following human instructions to explore and search for a specified target in an unfamiliar environment is a crucial skill for mobile service robots. Most of the previous works on object goal navigation have typically focused on a single input modality as the target, which may lead to limited consideration of language descriptions containing detailed attributes and spatial relationships. To address this limitation, we propose VLN-Game, a novel zero-shot framework for visual target navigation that can process object names and descriptive language targets effectively. To be more precise, our approach constructs a 3D object-centric spatial map by integrating pre-trained visual-language features with a 3D reconstruction of the physical environment. Then, the framework identifies the most promising areas to explore in search of potential target candidates. A game-theoretic vision-language model is employed to determine which target best matches the given language description. Experiments conducted on the Habitat-Matterport 3D (HM3D) dataset demonstrate that the proposed framework achieves state-of-the-art performance in both object goal navigation and language-based navigation tasks. Moreover, we show that VLN-Game can be easily deployed on real-world robots. The success of VLN-Game highlights the promising potential of using game-theoretic methods with compact vision-language models to advance decision-making capabilities in robotic systems.

![framework](img/framework.png)



## Installation

The code has been tested only with Python 3.10, CUDA 12.1. We also provide docker containers to run the Object-Goal Navigation task, or install all the packages manually to run all tasks.

### Installing Dependencies
- Setup the conda environment as Python3.10 + CUDA12.1 + PyTorch2.1.0.
    ```
    conda create -n vlmgame anaconda python=3.10
    conda activate vlmgame
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    ```
- We use 0.2.1 versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) as specified below:

- Installing habitat-sim:
    ```
    git clone https://github.com/facebookresearch/habitat-sim.git
    cd habitat-sim; git checkout tags/v0.2.1; 
    pip install -r requirements.txt; 
    python setup.py install --headless
    python setup.py install # (for Mac OS)
    ```

- Installing habitat-lab:
    ```
    git clone https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab; git checkout tags/v0.2.1; 
    pip install -e .
    ```

- Install Open3D for visualization: 

    ```
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    ```

- Install [Ground-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything):
    ```
    # Grounded-SAM
    export AM_I_DOCKER=False
    export BUILD_WITH_CUDA=True
    export CUDA_HOME=/path/to/anaconda3/envs/vlmgame/ #if you use the cluster to load the CUDA, you can check the path by "module show CUDAxxx"

    git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
    cd Grounded-Segment-Anything
    python -m pip install -e segment_anything
    python -m pip install -e GroundingDINO # delete pyproject.toml if building failure due to the cuda verson mismatch (link: https://github.com/vllm-project/vllm/issues/129#issuecomment-1805088950)

    #download
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

    ```

- Clone the repository and install other requirements:
    ```
    git clone https://github.com/ybgdgh/VLN-Game
    cd VLM-Game
    pip install -r requirements.txt
    ```


### Prepare Datasets

Create a `data` folder in `VLM-Game` path
```
mkdir data
cd data
```

#### Habitat Matterport
Download [HM3D_v0.1](https://aihabitat.org/datasets/hm3d/) dataset using download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d):
```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_val_habitat
```

#### Download Language description datasets

Download the [vlobjectnav_hm3d](https://drive.google.com/file/d/1fhXwBuGUOhF2jjW0ThtE_6rh_P3YQClj/view?usp=sharing) and unzip it in data/datasets path.



#### Setting up datasets
The code requires the datasets in a `data` folder in the following format:
```
VLM-Game/
  data/
    scene_datasets/
    versioned_data/
    datasets/
        objectgoal_hm3d/
        vlobjectnav_hm3d/
```

The `objectgoal_hm3d` is used for object-goal navigation, and `vlobjectnav_hm3d` is used for language descriptive target navigation.




## Run VLM_Game: 

The following commands should be run in the conda environemnt:
```
export GSA_PATH=/path/to/Grounded-Segment-Anything
```

#### Object-goal navigation
For evaluating the object-goal navigation task, run:
```
python main_vis.py
```
For multiprocessing, run:
```
python main_vis_vec.py
```

You can also add `-v 1` to enable the Open3D visualization UI to check the maps, and `-n 1` to set the number of multiprocessing.

![simulation](img/sim.gif)

#### Language-goal navigation
For evaluating the language-goal navigation task, the OpenAI API should be prepared firstly, then run:
```
python main_vln.py --api xx-xxxx
```
For multiprocessing, run:
```
python main_vln_vec.py --api xx-xxxx
```


#### Docker Setup
We provide docker containers for object-goal navigation task. This works on machines with an NVIDIA GPU and requires users to install nvidia-docker. To setup the habitat stack using docker follow the below steps:

1. Install Docker and pull the vlmgame docker image:
    ```
    docker pull ybgsdu/vln_llmgame:objnav-yolo-stair
    ```
2. Install NVIDIA Container Toolkit
    ```
    # Set up the package repository and the GPG key:
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    #Install the NVIDIA Docker toolkit:
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2

    #Restart the Docker daemon to load the changes:
    sudo systemctl restart docker
    ```
3. Mount the `data` folder into the docker when you run the docker
   ```
   sudo docker run --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all  -e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix -e LIBGL_ALWAYS_SOFTWARE=1 --net=host -it -v /path/to/data/:/root/VLN-GPT/data --gpus all ybgsdu/vln_llmgame:objnav-dino-stair /bin/bash -c "cd /root/VLN-GPT/ && ./setup_obj.sh 1 0"
    ```
    Make sure to adjust the last two parameters `1` and `0`  based on your own enrionment for the number of multiprocessing and the GPU_ID, respectively.

Since the language goal navigation relies on the rendering of Open3D, which can't run in this docker containers, we only run language goal navigation task locally.

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```
@article{yu2024vlngame,
  title={VLN-Game: Vision-Language Equilibrium Search for Zero-Shot Semantic Navigation}, 
  author={Bangguo Yu, Yuzhen Liu, Lei Han, Hamidreza Kasaei, Tingguang Li, and Ming Cao},
  journal={arXiv:2411.11609},
  year={2024}
}
```
