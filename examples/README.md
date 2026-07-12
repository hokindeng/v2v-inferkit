# Examples

This folder is a ready-to-use questions directory. Point run.py at it:

```bash
python3 run.py --model runway-aleph-v2v --questions-dir examples --output-dir ./outputs
```

## Layout

A questions directory contains one folder per domain (named `<domain>_task`),
and each domain folder contains one folder per task:

```
examples/                       <- the questions dir (--questions-dir)
└── turntable_task/             <- domain: turntable
    └── turntable_0000/         <- one task
        ├── prompt.txt          <- the edit instruction (required)
        ├── first_video.mp4     <- conditioning input video (required)
        └── first_frame.png     <- optional
```

Outputs land at `<output-dir>/<model>/turntable_task/turntable_0000.mp4`.

The sample task comes from VBVR-InferKit's sample runs: a turntable carries a
cyan cube behind a partial screen; the model must keep the cube present
through a full rotation. It is small on purpose (32 KB video) — one cheap
real-API smoke test.
