# wan-vace-14b-v2v — turntable_0000 smoke test

Verified 2026-07-15 on 1× L40S 48GB (g6e.xlarge + 48G NVMe swap for the
cpu-offload host-RAM footprint).

- Input: `examples/turntable_task/turntable_0000/` (prompt + first_video.mp4)
- Command: `V2V_WEIGHTS_DIR=~/models python3 run.py --model wan-vace-14b-v2v --questions-dir examples --output-dir ./outputs`
- Timing: 50 denoise steps @ ~26.8s/step; 37m38s end-to-end incl. model load
- Result: **pass** — output mirrors the conditioning video's motion profile
  (verified with ffmpeg freezedetect: the source is a step-wise turntable, and
  the output freezes/moves in lockstep with it) and the cyan cube persists
  through the occluded rotation.
- Settings: 832×480, 81 frames (input trimmed 135→81, VACE max), all-white
  mask (full repaint conditioned on source), UniPC flow_shift=3.0, cfg 5.0,
  seed 42, output 16 fps.
