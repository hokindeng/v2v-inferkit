# Models

The catalog contains 35 video-to-video models: nineteen commercial APIs and
sixteen local/open-weight integrations. Every wrapper consumes the task prompt
plus `video_path` and returns the standard eight fields (`success`,
`video_path`, `error`, `duration_seconds`, `generation_id`, `model`, `status`,
and `metadata`).

Old and new versions remain separately addressable while their APIs are live.
The catalog uses one hosted route per model and prefers fal.ai so one `FAL_KEY`
covers every commercial integration except Runway Aleph.

## Commercial APIs

### Runway

`runway-aleph-v2v` uses `RUNWAYML_API_SECRET`; fal.ai does not currently expose
the Aleph endpoint. Inputs under 2 seconds are padded before submission
(front-padded by cloning the first frame, same as the fal adapter). Prompts
over 1000 characters are refused, never truncated. `seed` is forwarded; the
task id is kept on every failure and timeout (`RunwayTaskError`).

### Shared fal.ai integrations

All other commercial models require only `FAL_KEY` and use the shared fal
adapter. The adapter uploads the source, submits a queued request, records the
fal request ID, downloads the result, and normalizes provider errors. It
rejects over-limit inputs before submission instead of allowing a provider to
silently truncate a paid benchmark run.

| Model ID | fal endpoint | Default and benchmark guard |
|---|---|---|
| `kling-v2-6-v2v` | `fal-ai/kling-video/o1/video-to-video/edit` | Legacy catalog ID for Kling O1 Edit; source 3–10s; original audio retained. |
| `luma-ray-3.2-v2v` | `luma/agent/ray/v3.2/video-to-video` | 720p, 5s, automatic edit controls. |
| `wan-2.7-video-edit` | `fal-ai/wan/v2.7/edit-video` | 1080p; source 2–10s; output duration follows the source. |
| `gemini-omni-flash-video-edit` | `google/gemini-omni-flash/edit` | Original Gemini Omni Flash editor. |
| `wan-3.0-video-edit` | `alibaba/wan-3.0/reference-to-video` | 720p; source ≤15s; output 2–30s. |
| `wan-3.0-prime-video-edit` | `alibaba/wan-3.0-prime/reference-to-video` | 720p; source ≤15s; output 2–30s. |
| `minimax-h3-v2v` | `minimax/h3/reference-to-video` | 768P; source 2–15s; output 5–15s. |
| `seedance-2.0-v2v` | `bytedance/seedance-2.0/reference-to-video` | 720p; source 2–15s; output 4–15s. |
| `seedance-2.0-fast-v2v` | `bytedance/seedance-2.0/fast/reference-to-video` | Same schema as 2.0, lower-latency tier. |
| `seedance-2.0-mini-v2v` | `bytedance/seedance-2.0/mini/reference-to-video` | Same reference limits, lower-cost Mini tier. |
| `seedance-2.5-v2v` | `bytedance/seedance-2.5/reference-to-video` | 720p; benchmark source/output ≤30s. |
| `gemini-omni-flash-1.1-video-edit` | `google/gemini-omni-flash/v1.1/edit` | 720p; fal does not currently publish an input-duration limit. |
| `kling-o3-pro-video-edit` | `fal-ai/kling-video/o3/pro/video-to-video/edit` | Source 3–15s; original audio retained. |
| `happy-horse-1.0-video-edit` | `alibaba/happy-horse/video-edit` | 720p; original audio retained; benchmark guard ≤15s because output is capped at 15s. |
| `grok-imagine-video-edit` | `xai/grok-imagine-video/edit-video` | 720p; source ≤8s because the API otherwise truncates it. |
| `grok-imagine-video-extend` | `xai/grok-imagine-video/extend-video` | Continuation: `duration` = extension seconds, integer 2–10, derived from `ground_truth.mp4` (rounded up; refused when no ground truth). The returned file is **source + extension stitched together** — trim the source span before scoring. Source 2–15s MP4; input seconds are billed too. |
| `veo-3.1-extend` | `fal-ai/veo3.1/extend-video` | Continuation with a **fixed 7 s / 720p** output (fal schema constants); audio off. Source ≤8s, 16:9 or 9:16. fal advertises it for Veo-made clips — verify on one question before a batch. |
| `ltx-2.3-extend` | `fal-ai/ltx-2.3/extend-video` | Continuation, LTX-2.3 Pro: `duration` is a float 2–20 s of new content and follows `ground_truth.mp4` exactly; `context` = the whole source; `mode: end`. Output reports duration/fps/frames. |

For Wan, MiniMax, and Seedance reference endpoints, output duration defaults to
the source duration rounded up to the next whole second and constrained to the
model range (a 2.5 s source therefore gets 3 s from Wan, 4 s from Seedance and
5 s from MiniMax — trim before scoring). Runtime wrapper kwargs (`run.py
--seed / --duration / --control`) may override `resolution`, `duration`,
`aspect_ratio`, `seed`, audio controls, and supported provider flags.

**The prompt is sent verbatim.** No reference labels, no "apply this edit"
framing, and Wan's server-side prompt expansion is off — the text in
`prompt.txt` is exactly what the model reads. Audio generation is off on every
profile that has the switch (silent benchmark sources; audio is billed extra).

Every fal request is traceable: the request id, the full payload and the prompt
sent are returned in the result metadata and written to the task's record by
`run.py`. A job that exceeds `max_wait` is cancelled on fal before the error is
raised; a job fal reports as failed (`Completed(error=...)`) raises
`FalRequestFailed` with its id. Downloads go through a `.part` file and are
ffprobe-validated before they count.

The official endpoint pages are the source of truth for changing limits and
pricing: [Kling O1](https://fal.ai/models/fal-ai/kling-video/o1/video-to-video/edit),
[Luma Ray 3.2](https://fal.ai/models/luma/agent/ray/v3.2/video-to-video),
[Wan 2.7](https://fal.ai/models/fal-ai/wan/v2.7/edit-video),
[Gemini Omni](https://fal.ai/models/google/gemini-omni-flash/edit),
[Wan 3.0](https://fal.ai/models/alibaba/wan-3.0/reference-to-video),
[Wan Prime](https://fal.ai/models/alibaba/wan-3.0-prime/reference-to-video),
[MiniMax H3](https://fal.ai/minimax-h3),
[Seedance 2.0](https://fal.ai/models/bytedance/seedance-2.0/reference-to-video),
[Seedance Fast](https://fal.ai/models/bytedance/seedance-2.0/fast/reference-to-video),
[Seedance Mini](https://fal.ai/models/bytedance/seedance-2.0/mini/reference-to-video),
[Seedance 2.5](https://fal.ai/models/bytedance/seedance-2.5/reference-to-video),
[Gemini Omni 1.1](https://fal.ai/models/google/gemini-omni-flash/v1.1/edit),
[Kling O3](https://fal.ai/models/fal-ai/kling-video/o3/pro/video-to-video/edit),
[Happy Horse](https://fal.ai/models/alibaba/happy-horse/video-edit),
[Grok Edit](https://fal.ai/docs/model-api-reference/video-generation-api/xai-grok-imagine-video), and
[Grok Extend](https://fal.ai/models/xai/grok-imagine-video/extend-video).

## Local/open-weight integrations

All local integrations use `envs/<model-id>/` and checkpoints below
`V2V_WEIGHTS_DIR` (default `weights/`). Install one with
`bash setup/install_model.sh --model <model-id>` or install all with `--local`.
The former `--opensource` flag remains an alias.

### True video editing

| Model ID | Upstream execution path | Important behavior |
|---|---|---|
| `wan-vace-1.3b-v2v` | Diffusers `WanVACEPipeline` | 480p source-conditioned editing; lighter VACE checkpoint. |
| `wan-vace-14b-v2v` | Diffusers `WanVACEPipeline` | 480p; CPU offload by default, `V2V_NO_OFFLOAD=1` on 80GB GPUs. |
| `hy-omniweaving-v2v` | Official OmniWeaving editing task | 480p with component offload; includes gated FLUX dependency. |
| `joyai-video-edit-v2v` | Official FastAPI/WebSocket streaming server | Automatically starts a localhost service, uploads JPEG frames, finalizes its recorder, downloads the MP4, then stops it. `JOYAI_SERVER_URL` may point to a preloaded server launched with `--record-dir`. |
| `bernini-r-1.3b-v2v` | Official `infer_single_gpu.py` | Source-only `v2v` by default; optional `reference_image_path` switches to `rv2v`. |
| `bernini-r-14b-v2v` | Same Bernini wrapper, 14B checkpoint | H100-class GPU recommended by upstream. |
| `kiwi-edit-5b-v2v` | Diffusers remote pipeline | Instruction+reference checkpoint; references are used only when `reference_image_path` is explicitly supplied. |
| `editto-v2v` | Official Ditto DiffSynth script | VACE-14B plus Ditto LoRA; CC-BY-NC-SA-4.0 and non-commercial. |
| `sama-14b-v2v` | Official SAMA single-video CLI | Wan2.1-T2V-14B base plus semantic-editing checkpoint. |
| `omnivideo2-1.3b-v2v` | Official 1.3B E2E/VLM entrypoint | Requires Qwen3-VL-30B-A3B; component offload supported. |
| `omnivideo2-a14b-v2v` | Official A14B E2E/VLM entrypoint | Requires Qwen3-VL-30B-A3B and an 80GB-class GPU. |
| `coinve-edit-v2v` | Official single-video CoinVE CLI | A normal prompt is one instruction; callers may pass `instructions=[...]` for the native 2–5 instruction mode. |
| `lucy-edit-1.1-v2v` | Diffusers `LucyEditPipeline` | 5B editor with FP32 VAE; model license is non-commercial. |

`first_frame.png` is not silently treated as a reference image. This preserves
identical source-video benchmark semantics across models. Specialized callers
can pass `reference_image_path` directly to Bernini, Kiwi, or JoyAI wrappers.

### Other video-conditioned capabilities

| Model ID | Capability | Caveat |
|---|---|---|
| `ltx-2.3-dev-v2v` | `video_conditioned_generation` | IC-LoRA path; the official 2.3 V2V pipeline uses a distilled checkpoint. |
| `magi-24b-v2v` | `video_continuation` | Generates frames after a video prefix and does not edit the supplied footage. |
| `cosmos3-super-v2v` | `controlled_video_transfer` | Uses source-derived edge control rather than instruction-only editing. |

Catalog license metadata separates code and weight licenses and includes a
`commercial_use` value. `null` means upstream terms require manual review; it
does not mean commercial use is allowed.

## Operational notes

- These APIs cost real money and pricing changes frequently. Check the linked
  provider page before batch runs; automated tests and `--dry-run` do not make
  paid calls.
- fal-hosted inputs transit hosted storage. Do not submit sensitive
  material unless that data handling is acceptable.
- `ffprobe` is required for hosted-input validation and `ffmpeg` is required
  when a source needs minimum-duration padding. Padding is always added at the
  **front** by cloning the first frame: a benchmark clip ends where the
  continuation has to pick up, so the ending and the motion timing are never
  altered; only the opening hold gets longer.
- H3 Max, Happy Horse 1.1, and Grok Imagine Video 1.5 are not cataloged as V2V
  entries because their currently published endpoints do not accept a source
  video for editing. Add them when a real video-input endpoint becomes live.
