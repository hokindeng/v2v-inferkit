# Models

The catalog contains 21 video-to-video models: sixteen commercial APIs and
five local open-source integrations. Every wrapper consumes the task prompt
plus `video_path` and returns the standard eight fields (`success`,
`video_path`, `error`, `duration_seconds`, `generation_id`, `model`, `status`,
and `metadata`).

Old and new versions remain separately addressable while their APIs are live.
Where several hosts expose the same base model, the catalog keeps one hosted
entry and currently prefers fal.ai.

## Commercial APIs

### Existing provider integrations

| Model ID | Provider/key | Behavior |
|---|---|---|
| `runway-aleph-v2v` | Runway / `RUNWAYML_API_SECRET` | Aleph 2 video-to-video; inputs under 2 seconds are padded. |
| `kling-v2-6-v2v` | Kling / `KLING_API_KEY` or JWT pair, plus `FAL_KEY` for upload | Kling Omni editing; inputs under 4 seconds are padded. |
| `luma-ray-3.2-v2v` | Luma / `LUMA_AGENTS_API_KEY`, plus `FAL_KEY` for upload | Ray 3.2 `video_edit`, 720p auto controls. |
| `wan-2.7-video-edit` | WaveSpeed / `WAVESPEED_API_KEY` | Wan 2.7 prompt-driven editing; source is sent as a data URI. |
| `gemini-omni-flash-video-edit` | WaveSpeed / `WAVESPEED_API_KEY` | Original Gemini Omni Flash video editor; source is sent as a data URI. |

### Shared fal.ai integrations

All models below require `FAL_KEY` and use the shared fal adapter. The adapter
uploads the source, submits a queued request, records the fal request ID,
downloads the result, and normalizes provider errors. It rejects over-limit
inputs before submission instead of allowing a provider to silently truncate a
paid benchmark run.

| Model ID | fal endpoint | Default and benchmark guard |
|---|---|---|
| `wan-3.0-video-edit` | `alibaba/wan-3.0/reference-to-video` | 720p; source ≤15s; output 2–30s. |
| `wan-3.0-prime-video-edit` | `alibaba/wan-3.0-prime/reference-to-video` | 720p; source ≤15s; output 2–30s. |
| `minimax-h3-v2v` | `minimax/h3/reference-to-video` | 768P; source 2–15s; output 4–15s. |
| `seedance-2.0-v2v` | `bytedance/seedance-2.0/reference-to-video` | 720p; source 2–15s; output 4–15s. |
| `seedance-2.0-fast-v2v` | `bytedance/seedance-2.0/fast/reference-to-video` | Same schema as 2.0, lower-latency tier. |
| `seedance-2.0-mini-v2v` | `bytedance/seedance-2.0/mini/reference-to-video` | Same reference limits, lower-cost Mini tier. |
| `seedance-2.5-v2v` | `bytedance/seedance-2.5/reference-to-video` | 720p; benchmark source/output ≤30s. |
| `gemini-omni-flash-1.1-video-edit` | `google/gemini-omni-flash/v1.1/edit` | 720p; fal does not currently publish an input-duration limit. |
| `kling-o3-pro-video-edit` | `fal-ai/kling-video/o3/pro/video-to-video/edit` | Source 3–15s; original audio retained. |
| `happy-horse-1.0-video-edit` | `alibaba/happy-horse/video-edit` | 720p; original audio retained; benchmark guard ≤15s because output is capped at 15s. |
| `grok-imagine-video-edit` | `xai/grok-imagine-video/edit-video` | 720p; source ≤8s because the API otherwise truncates it. |

For Wan, MiniMax, and Seedance reference endpoints, output duration defaults to
the source duration rounded up to the next whole second and constrained to the
model range. Runtime wrapper kwargs may override `resolution`, `duration`,
`aspect_ratio`, `seed`, audio controls, and supported provider flags.

Reference labels are added only when the prompt does not already include one:
Wan and MiniMax use `Video 1`, Seedance 2.0 uses `@Video1`, Seedance 2.5 uses
`[Video1]`, and Kling O3 uses `@Video1`. The user's edit instruction is otherwise
left unchanged.

The official endpoint pages are the source of truth for changing limits and
pricing: [Wan 3.0](https://fal.ai/models/alibaba/wan-3.0/reference-to-video),
[Wan Prime](https://fal.ai/models/alibaba/wan-3.0-prime/reference-to-video),
[MiniMax H3](https://fal.ai/minimax-h3),
[Seedance 2.0](https://fal.ai/models/bytedance/seedance-2.0/reference-to-video),
[Seedance Fast](https://fal.ai/models/bytedance/seedance-2.0/fast/reference-to-video),
[Seedance Mini](https://fal.ai/models/bytedance/seedance-2.0/mini/reference-to-video),
[Seedance 2.5](https://fal.ai/models/bytedance/seedance-2.5/reference-to-video),
[Gemini Omni 1.1](https://fal.ai/models/google/gemini-omni-flash/v1.1/edit),
[Kling O3](https://fal.ai/models/fal-ai/kling-video/o3/pro/video-to-video/edit),
[Happy Horse](https://fal.ai/models/alibaba/happy-horse/video-edit), and
[Grok Edit](https://fal.ai/docs/model-api-reference/video-generation-api/xai-grok-imagine-video).

## Local open-source integrations

| Model ID | Behavior |
|---|---|
| `wan-vace-14b-v2v` | Wan2.1-VACE-14B true editing, 480p. |
| `hy-omniweaving-v2v` | HY-OmniWeaving editing, 480p with offload support. |
| `ltx-2.3-dev-v2v` | LTX-2.3 IC-LoRA conditioning; official V2V pipeline currently uses a distilled checkpoint. |
| `magi-24b-v2v` | Prefix-video continuation, not source-footage editing. |
| `cosmos3-super-v2v` | Edge-controlled video transfer. |

## Operational notes

- These APIs cost real money and pricing changes frequently. Check the linked
  provider page before batch runs; automated tests and `--dry-run` do not make
  paid calls.
- fal, Kling, and Luma inputs transit hosted storage. Do not submit sensitive
  material unless that data handling is acceptable.
- `ffprobe` is required for hosted-input validation and `ffmpeg` is required
  when a source needs minimum-duration padding.
- H3 Max, Happy Horse 1.1, and Grok Imagine Video 1.5 are not cataloged as V2V
  entries because their currently published endpoints do not accept a source
  video for editing. Add them when a real video-input endpoint becomes live.
