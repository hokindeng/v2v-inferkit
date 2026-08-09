# v2v-inferkit

Unified inference toolkit for **video-to-video** generation models.
Give it a benchmark task (a conditioning video plus a text prompt) and it runs
any of ten V2V models — five commercial APIs and five open-source models —
behind one CLI.

## Commercial API models

| Model | Provider | API key |
|---|---|---|
| runway-aleph-v2v | Runway Aleph 2 | RUNWAYML_API_SECRET |
| kling-v2-6-v2v | Kling Omni | KLING_API_KEY + FAL_KEY |
| luma-ray-3.2-v2v | Luma Ray 3.2 video_edit | LUMA_AGENTS_API_KEY + FAL_KEY |
| wan-2.7-video-edit | WAN 2.7 via WaveSpeed | WAVESPEED_API_KEY |
| gemini-omni-flash-video-edit | Gemini Omni via WaveSpeed | WAVESPEED_API_KEY |

No GPU, weights, or per-model venv needed — just API keys in `.env`.

## Open-source models (local GPU)

| Model | Flagship checkpoint | GPU needed | Notes |
|---|---|---|---|
| wan-vace-14b-v2v | Wan2.1-VACE-14B (diffusers) | 1× 48GB, 480p | true V2V editing |
| hy-omniweaving-v2v | tencent/HY-OmniWeaving | 1× ≥14GB w/ offload, 480p | true V2V editing; model dir needs 4 extra components (one HF-gated) |
| ltx-2.3-dev-v2v | LTX-2.3 22B via IC-LoRA | 1× 48GB (fp8) / 80GB (bf16) | ⚠ official v2v pipeline is distilled-only — pending team ruling |
| magi-24b-v2v | MAGI-1 24B base | 4× 80GB | ⚠ v2v = video CONTINUATION, not editing |
| cosmos3-super-v2v | nvidia/Cosmos3-Super | 4× 80GB, 132GB ckpt | video transfer (edge control) |

Install one with:

```bash
bash setup/install_model.sh --model wan-vace-14b-v2v
# reuse an existing checkpoint dir:
V2V_WEIGHTS_DIR=~/models bash setup/install_model.sh --model wan-vace-14b-v2v
```

Each open-source model runs in its own venv (`envs/<model>/`); the runner
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
ffmpeg must be on PATH (used to pad too-short input videos).
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
- `v2vinferkit/models/*.py` — per-provider wrappers. Each returns the same
  8-field result dict and routes to its v2v path when a video_path kwarg is
  present.
- `v2vinferkit/runner/inference.py` — dispatch. Commercial API models load
  in-process; open-source models run in their model-specific venv via a
  subprocess worker.

See docs/MODELS.md for per-model behavior and docs/ADDING_MODELS.md for how
to add one.
