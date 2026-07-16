# hy-omniweaving-v2v — turntable_0000 smoke test

Verified runs 2026-07-15 on 1× L40S 48GB (g6e.xlarge + 48G NVMe swap; the
33GB fp32 transformer loads through host RAM).

- Input: `examples/turntable_task/turntable_0000/` (prompt + first_video.mp4)
- Command: `V2V_WEIGHTS_DIR=~/models python3 run.py --model hy-omniweaving-v2v --questions-dir examples --output-dir ./outputs`
- Timing: 50 denoise steps @ ~22.7s/step (first step ~155s warm-up); ~31 min
  end-to-end incl. model load. flash-attn not installed → torch SDPA fallback
  (expected, warnings are benign).
- Result: **pipeline pass, quality caveat** — the output faithfully follows
  the source scene, camera and motion, but the cyan cube itself is rendered
  as a faint gray smudge (object dropped). Relevant signal for an
  object-permanence benchmark rather than an integration bug; possibly
  aggravated by the model's 480p-only resolution (small object).
- Settings: --task editing, 480p, 81 frames, aspect 16:9, seed 0,
  --group_offloading false (see wrapper comment), output fps follows input (24).
