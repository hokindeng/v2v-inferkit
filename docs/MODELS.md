# Models

Five video-to-video models, all commercial APIs. Every wrapper takes the task
prompt plus a video_path and returns the standard 8-field result dict
(success, video_path, error, duration_seconds, generation_id, model, status,
metadata).

## runway-aleph-v2v (Runway Aleph 2)

- Key: RUNWAYML_API_SECRET. Official runwayml SDK, video_to_video.create.
- Input video uploaded through Runway's own ephemeral upload — no CDN key needed.
- Duration is intentionally not sent (Aleph rejects it). Inputs shorter than
  2 seconds are padded with ffmpeg before upload.

## kling-v2-6-v2v (Kling Omni)

- Keys: KLING_API_KEY (or KLING_ACCESS_KEY + KLING_SECRET_KEY for JWT signing,
  needs PyJWT) plus FAL_KEY.
- Input video uploaded to the fal CDN, then POSTed to the omni-video endpoint
  with model kling-video-o1, pro mode. Minimum input duration 4 s (ffmpeg pad).

## luma-ray-3.2-v2v (Luma Ray 3.2 video_edit)

- Keys: LUMA_AGENTS_API_KEY (LUMA_API_KEY also accepted) plus FAL_KEY.
- Input uploaded to the fal CDN; generation type video_edit at 720p with
  auto controls.

## wan-2.7-video-edit and gemini-omni-flash-video-edit (WaveSpeed)

- Key: WAVESPEED_API_KEY for both.
- Input video is inlined as a base64 data URI — no upload step, no CDN key.
- Both are prompt-driven video editors; wan-2.7 supports 720p/1080p.

## Notes

- These APIs cost real money — figure roughly $1-3 per generation depending
  on the model. Budget before batch runs.
- Kling and Luma inputs transit the fal CDN; Runway inputs transit Runway's
  storage. Don't send sensitive material through these models.
