#!/usr/bin/env python3
"""Generic subprocess worker for v2v-inferkit model inference.

This script runs inside a model-specific virtual environment.
It imports the model wrapper (which may have heavy dependencies like
torch, diffusers, etc.) and runs inference, returning results as JSON
on stdout.

--image-path is optional (v2v tasks may have no first_frame.png; the
conditioning video arrives as video_path inside --kwargs-json), and the
result marker is __V2V_RESULT__.

Usage:
    /path/to/envs/<model>/bin/python _subprocess_worker.py \
        --model-name wan-vace-14b-v2v \
        --prompt "Edit instruction" \
        --output-dir /path/to/output \
        --kwargs-json '{"video_path": "/path/to/first_video.mp4"}'
"""

import sys
import json
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="v2v-inferkit subprocess worker")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kwargs-json", default="{}")
    args = parser.parse_args()

    # Ensure v2v-inferkit root is on sys.path
    kit_root = str(Path(__file__).parent.parent.parent)
    if kit_root not in sys.path:
        sys.path.insert(0, kit_root)

    start_time = time.time()

    # Import catalog (light - no heavy deps)
    from v2vinferkit.runner.MODEL_CATALOG import AVAILABLE_MODELS

    if args.model_name not in AVAILABLE_MODELS:
        result = {
            "success": False,
            "video_path": None,
            "error": f"Unknown model: {args.model_name}",
            "duration_seconds": time.time() - start_time,
            "generation_id": None,
            "model": args.model_name,
            "status": "failed",
            "metadata": {},
        }
        print("__V2V_RESULT__" + json.dumps(result, default=str))
        return

    config = AVAILABLE_MODELS[args.model_name]

    # Dynamically import the wrapper module (this triggers heavy deps in venv)
    import importlib

    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])

    # Build init kwargs
    init_kwargs = {
        "model": config["model"],
        "output_dir": args.output_dir,
    }
    if "args" in config:
        init_kwargs.update(config["args"])

    # Instantiate wrapper and run inference
    wrapper = wrapper_class(**init_kwargs)

    kwargs = json.loads(args.kwargs_json)
    result = wrapper.generate(args.image_path, args.prompt, **kwargs)

    # Ensure result is JSON-serializable
    serializable_result = {}
    for key, value in result.items():
        try:
            json.dumps(value, default=str)
            serializable_result[key] = value
        except (TypeError, ValueError):
            serializable_result[key] = str(value)

    # Output result as JSON on stdout (last line)
    print("__V2V_RESULT__" + json.dumps(serializable_result, default=str))


if __name__ == "__main__":
    main()
