"""OmniVideo2 1.3B/A14B V2V editing via the official E2E inference entrypoint."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, run_command, success_result, weights_path


class OmniVideo2Service:
    def __init__(self, model: str, task: str):
        self.model = model
        self.task = task
        self.repo = repo_path("Omni-Video", "OMNIVIDEO2_REPO_PATH")
        checkpoint_name = model.rsplit("/", 1)[-1]
        self.checkpoint = Path(os.environ.get("OMNIVIDEO2_WEIGHTS_PATH") or str(weights_path(checkpoint_name)))
        self.qwen = Path(os.environ.get("OMNIVIDEO2_QWEN_PATH") or str(weights_path("Qwen3-VL-30B-A3B-Instruct")))

    def generate_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        output_path: Path,
        *,
        num_frames: int = 41,
        size: str = "832*480",
        num_inference_steps: int = 40,
        seed: int = 42,
        fps: int = 8,
        sampling_rate: int = 3,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        repo = require_dir(self.repo, "OmniVideo2 repository")
        checkpoint = require_dir(self.checkpoint, "OmniVideo2 checkpoint")
        qwen = require_dir(self.qwen, "Qwen3-VL checkpoint")
        source = require_file(video_path, "source video")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        num_frames = max(1, min(int(num_frames), 81))
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        sample_id = f"v2vinferkit_{time.time_ns()}"
        expected = repo / "outputs" / f"{source.stem}_id{sample_id}_edited.mp4"
        entrypoint = (
            repo / "tools" / "inference" / "generate_omni_v2v_1_3B.py"
            if "1.3B" in self.task
            else repo / "tools" / "inference" / "generate_omni_v2v.py"
        )
        require_file(entrypoint, "OmniVideo2 inference entrypoint")

        with tempfile.TemporaryDirectory(prefix="omnivideo2_v2v_") as tmp:
            prompt_file = Path(tmp) / "input.jsonl"
            prompt_file.write_text(
                json.dumps({"id": sample_id, "prompt": prompt, "source_clip_path": str(source)}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                entrypoint,
                "--task", self.task,
                "--size", size,
                "--frame_num", str(num_frames),
                "--ckpt_dir", checkpoint,
                "--prompt_file", prompt_file,
                "--qwen3vl_model_path", qwen,
                "--base_seed", str(seed),
                "--sample_steps", str(num_inference_steps),
                "--sample_fps", str(fps),
                "--sampling_rate", str(sampling_rate),
            ]
            run_command(command, cwd=repo, timeout=timeout)
            generated = require_file(expected, "OmniVideo2 output video")
            shutil.copy2(generated, output_path)
            generated.unlink()
        return {"task": self.task, "checkpoint": str(checkpoint), "qwen_checkpoint": str(qwen), "num_frames": num_frames, "size": size, "num_inference_steps": num_inference_steps, "seed": seed, "fps": fps}


class OmniVideo2Wrapper(ModelWrapper):
    def __init__(self, model: str, output_dir: str = "./outputs", task: str = "v2v-A14B", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = OmniVideo2Service(model, task)

    def generate(self, image_path, text_prompt, duration=5.0, output_filename=None, video_path=None, **kwargs):
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "video_path is required")
        kwargs.pop("question_data", None)
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            metadata = self.service.generate_video(video_path, text_prompt, output, **kwargs)
        except Exception as exc:
            return failed_result(self.model, text_prompt, start, exc, video_path)
        return success_result(self.model, text_prompt, start, output, video_path, "omnivideo2", metadata)
