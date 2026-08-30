# v2v-inferkit

Unified inference toolkit for **video-to-video** generation models.
Give it a benchmark task (a conditioning video plus a text prompt) and it runs
any of 35 V2V models — nineteen commercial APIs and sixteen local/open-weight models —
behind one CLI.

## Commercial API models

| Model | Provider | API key |
|---|---|---|
| runway-aleph-v2v | Runway Aleph 2 | RUNWAYML_API_SECRET |
| kling-v2-6-v2v | Kling O1 Edit via fal.ai | FAL_KEY |
| luma-ray-3.2-v2v | Luma Ray 3.2 via fal.ai | FAL_KEY |
| wan-2.7-video-edit | WAN 2.7 via fal.ai | FAL_KEY |
| wan-3.0-video-edit | WAN 3.0 via fal.ai | FAL_KEY |
| wan-3.0-prime-video-edit | WAN 3.0 Prime via fal.ai | FAL_KEY |
| gemini-omni-flash-video-edit | Gemini Omni via fal.ai | FAL_KEY |
| gemini-omni-flash-1.1-video-edit | Gemini Omni Flash 1.1 via fal.ai | FAL_KEY |
| minimax-h3-v2v | MiniMax H3 reference-to-video via fal.ai | FAL_KEY |
| seedance-2.0-v2v | Seedance 2.0 via fal.ai | FAL_KEY |
| seedance-2.0-fast-v2v | Seedance 2.0 Fast via fal.ai | FAL_KEY |
| seedance-2.0-mini-v2v | Seedance 2.0 Mini via fal.ai | FAL_KEY |
| seedance-2.5-v2v | Seedance 2.5 via fal.ai | FAL_KEY |
| kling-o3-pro-video-edit | Kling O3 Pro Edit via fal.ai | FAL_KEY |
| happy-horse-1.0-video-edit | Happy Horse 1.0 Edit via fal.ai | FAL_KEY |
| grok-imagine-video-edit | Grok Imagine Video Edit via fal.ai | FAL_KEY |
| grok-imagine-video-extend | Grok Imagine Video Extend (continuation) via fal.ai | FAL_KEY |
| veo-3.1-extend | Veo 3.1 Extend (continuation, fixed 7 s) via fal.ai | FAL_KEY |
| ltx-2.3-extend | LTX-2.3 Pro Extend (continuation, 2–20 s) via fal.ai | FAL_KEY |

No GPU, weights, or per-model venv needed. One `FAL_KEY` runs every fal-hosted
model in the table; Runway alone uses `RUNWAYML_API_SECRET` because its Aleph
endpoint is not available through fal.ai.

## Configure all API keys

Export every key with one shell command before running inference:

```bash
export FAL_KEY="your-fal-key" \
       RUNWAYML_API_SECRET="your-runway-key" \
       HF_TOKEN="your-huggingface-token"
```

`FAL_KEY` covers all fal-hosted commercial models, `RUNWAYML_API_SECRET` is only
for Runway Aleph, and `HF_TOKEN` is only for gated local checkpoints. If you
only run fal-hosted models, `export FAL_KEY="your-fal-key"` is all you need.

Shell exports last for the current terminal session. To load the same keys
automatically in future sessions, put the command in `~/.zshrc` (zsh) or
`~/.bashrc` (bash), then open a new terminal.

Alternatively, keep the keys project-local:

```bash
cp env.template .env
# Edit .env and replace the empty values; run.py loads it automatically.
```

## Local/open-weight models (local GPU)

### True video editing

| Model | Checkpoint | GPU / resolution | License notes |
|---|---|---|---|
| wan-vace-1.3b-v2v | Wan2.1-VACE-1.3B Diffusers | 1 GPU, 480p | Apache-2.0 |
| wan-vace-14b-v2v | Wan2.1-VACE-14B Diffusers | 1×48GB, 480p | Apache-2.0 |
| hy-omniweaving-v2v | tencent/HY-OmniWeaving | 1×≥14GB with offload, 480p | Tencent Hunyuan license |
| joyai-video-edit-v2v | jdopensource/JoyAI-Video-Edit | 1×32GB reference, 840×480 | Apache-2.0 |
| bernini-r-1.3b-v2v | ByteDance/Bernini-R-1.3B-Diffusers | 1 GPU | Apache-2.0 |
| bernini-r-14b-v2v | ByteDance/Bernini-R-Diffusers | H100-class recommended | Apache-2.0 |
| kiwi-edit-5b-v2v | Kiwi-Edit 5B instruct+reference | 1 GPU, 480p | MIT code; checkpoint license unspecified |
| editto-v2v | Ditto VACE-14B LoRA | 1 GPU, 480p | **CC-BY-NC-SA-4.0, non-commercial** |
| sama-14b-v2v | syxbb/SAMA-14B | high-memory GPU, 480p | Apache-2.0 |
| omnivideo2-1.3b-v2v | Fudan-FUXI/OmniVideo2-1.3B | 1 GPU with offload, 480p | upstream license unspecified |
| omnivideo2-a14b-v2v | Fudan-FUXI/OmniVideo2-A14B | 1×80GB recommended, 480p | upstream license unspecified |
| coinve-edit-v2v | FireCRT/CoinVE-Edit | ~54GB at 720p/49f | Apache-2.0 weights |
| lucy-edit-1.1-v2v | decart-ai/Lucy-Edit-1.1-Dev | 1 GPU with offload, 480p | **non-commercial model license** |

### Other video-conditioned capabilities

| Model | Capability | GPU needed | Caveat |
|---|---|---|---|
| ltx-2.3-dev-v2v | video-conditioned generation via IC-LoRA | 1×48GB fp8 / 80GB bf16 | official 2.3 V2V path is distilled-checkpoint based |
| magi-24b-v2v | video continuation | 4×80GB | extends the prefix; does not edit source frames |
| cosmos3-super-v2v | controlled video transfer | 4×80GB, 132GB checkpoint | edge control derived from source video |

Install one with:

```bash
bash setup/install_model.sh --model wan-vace-14b-v2v
# install every local integration (large downloads):
bash setup/install_model.sh --local
# reuse an existing checkpoint dir:
V2V_WEIGHTS_DIR=~/models bash setup/install_model.sh --model wan-vace-14b-v2v
```

Each local model runs in its own venv (`envs/<model>/`); the runner
detects the venv and dispatches inference to it in a subprocess automatically.
Research notes per model (entrypoints, VRAM, gotchas) live with the
integration task owner.

## Quick start

```bash
pip install -e .
cp env.template .env        # fill in the keys you need
python3 run.py --list-models
python3 run.py --model runway-aleph-v2v \
    --questions-dir examples --output-dir ./outputs
```

Add `--dry-run` to see the planned jobs without calling any API.

Benchmark controls: `--seed N` (forwarded to every endpoint that accepts one),
`--duration S` (endpoint-checked override of the output/extension length),
`--max-wait S` (cancel a hosted job after S seconds), `--control key=value`
(any endpoint control from the profile, repeatable), `--workers N` (concurrent
generations per model), `--retries N` (transient errors only), `--retry-failed`
(rerun only tasks with a `.failed.json`).

Every generation leaves a record next to its video: `<task_id>.json` on
success (request/task id, the exact payload and prompt sent, input padding,
output geometry) or `<task_id>.failed.json` on failure, plus one
`run-<timestamp>.json` manifest per model. A task is only skipped on rerun when
its video decodes and its success record exists.
ffmpeg must be on PATH (used to front-pad too-short input videos by cloning
their first frame — the clip's ending is never touched).
Run run.py from the repo root — it is not installed as a console script.

## Task layout

`--questions-dir` points at a directory with one folder per domain and one
subfolder per task. The shipped `examples/` directory is a working instance:

```
examples/                       <- the questions dir
└── turntable_task/             <- one domain
    └── turntable_0000/         <- one task
        ├── prompt.txt          # text prompt (required)
        ├── first_video.mp4     # conditioning input video (required)
        ├── first_frame.png     # optional reference image
        └── ground_truth.mp4    # optional; its frame count is passed to
                                #   the model as num_frames
```

A task runs as long as it has prompt.txt and first_video.mp4. final_frame.png
is also picked up if present. The `_task` suffix on domain folders is
conventional, not required — any folder name works.

Output: `{output_dir}/{model}/{domain_folder}/{task_id}.mp4`. Existing
outputs are skipped unless you pass `--no-skip-existing`. Narrow a run with
`--task-id turntable_0000` or `--domains turntable_task`.

## How it works

- `v2vinferkit/runner/MODEL_CATALOG.py` — the registry: one entry per model
  with its wrapper class, modality, and dispatch settings.
- `v2vinferkit/models/*.py` — provider wrappers. The hosted fal endpoints share
  one profile-driven adapter; every wrapper returns the same 8-field result
  dict and routes to its v2v path when a video_path kwarg is present.
- `v2vinferkit/runner/inference.py` — dispatch. Commercial API models load
  in-process; local/open-weight models run in their model-specific venv via a
  subprocess worker.

See docs/MODELS.md for per-model behavior and docs/ADDING_MODELS.md for how
to add one.
