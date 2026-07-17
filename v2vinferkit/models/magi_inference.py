"""MAGI-1 24B video-to-video (continuation) inference (open-source, multi-GPU).

⚠️ SCOPE CAVEAT: MAGI's v2v mode is PREFIX-VIDEO CONTINUATION, not editing.
The first 32 frames of the conditioning video (after 24fps resample) are
frozen as clean context and replayed verbatim at the start of the output;
the prompt only steers the newly generated frames. It cannot restyle or
transform the conditioning footage (see SandAI-org/MAGI-1 issue #47). Whether
that counts as V2V for the benchmark is a team-level decision — this wrapper
exists so the model can be evaluated either way.

Checkpoint: MAGI-1-24B base (bf16). NOT MAGI-1.1-24B base — that checkpoint is
currently incompatible with the public inference code (issue #123, packed qkv
keys). NOT the 4.5B/distill/fp8 variants (team flagship-only rule; fp8 also
doesn't run on Ampere).

Hardware: 8xH100 recommended; 4xA100-80GB works per maintainer (issue #97)
with pp_size=1, cp_size=<n_gpus>, cp_strategy=cp_ulysses (cp_shuffle_overlap
silently produces garbage on the base model). ~46GB/GPU weights after load —
80GB-class GPUs mandatory. Does NOT run on a single 48GB GPU.

Runs inside its model venv (envs/magi-24b-v2v) via _subprocess_worker.py; the
wrapper itself shells out to torchrun in the MAGI repo clone.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper

# MAGI hardcodes 32 prefix frames (video_process.py); they are replayed at the
# start of the output and must be trimmed if only generated content is wanted.
_PREFIX_FRAMES = 32
_FPS = 24


def _kit_root() -> Path:
    return Path(__file__).parent.parent.parent


def _weights_dir() -> Path:
    return Path(os.environ.get("V2V_WEIGHTS_DIR", str(_kit_root() / "weights"))) / "MAGI-1"


def _repo_dir() -> Path:
    return _kit_root() / "repos" / "MAGI-1"


class MagiService:
    """Builds the run config and shells out to torchrun entry.py --mode v2v."""

    def __init__(self, model: str = "sand-ai/MAGI-1:24B_base"):
        self.model_id = model
        self.repo_path = Path(os.environ.get("MAGI_REPO_PATH", str(_repo_dir())))
        self.weights_path = Path(os.environ.get("MAGI_WEIGHTS_PATH", str(_weights_dir())))
        self.num_gpus = int(os.environ.get("MAGI_NUM_GPUS", "4"))

    def _build_config(self, workdir: Path) -> Path:
        """Instantiate the 24B base config with our paths and parallelism."""
        base_config = self.repo_path / "example" / "24B" / "24B_base_config.json"
        config = json.loads(base_config.read_text())
        # settings live in NESTED sections (runtime_config / engine_config) —
        # top-level keys are silently ignored by MAGI (hit 2026-07-17)
        rt, eng = config["runtime_config"], config["engine_config"]
        rt["load"] = str(self.weights_path / "ckpt" / "magi" / "24B_base")
        rt["t5_pretrained"] = str(self.weights_path / "ckpt" / "t5")
        rt["vae_pretrained"] = str(self.weights_path / "ckpt" / "vae")
        eng["pp_size"] = 1
        eng["cp_size"] = self.num_gpus
        eng["cp_strategy"] = "cp_ulysses"  # cp_shuffle_overlap = garbage on base (issue #97)
        rt["num_steps"] = int(os.environ.get("MAGI_NUM_STEPS", "64"))  # 64 for base quality (issue #54)
        rt["fps"] = _FPS  # low fps crashes tiled VAE decode (issue #108)
        out = workdir / "run_config.json"
        out.write_text(json.dumps(config, indent=2))
        return out

    def _trim_prefix(self, raw_path: Path, final_path: Path):
        """Drop the replayed 32-frame prefix so only generated content remains."""
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-vf", f"select='gte(n,{_PREFIX_FRAMES})',setpts=N/{_FPS}/TB",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
                str(final_path),
            ],
            capture_output=True, check=True,
        )

    def generate_video(
        self,
        video_path: Union[str, Path],
        text_prompt: str,
        output_path: Path,
        keep_prefix: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        if not (self.repo_path / "inference" / "pipeline" / "entry.py").exists():
            raise FileNotFoundError(
                f"MAGI repo not found at {self.repo_path} — run "
                "setup/install_model.sh --model magi-24b-v2v"
            )

        workdir = Path(tempfile.mkdtemp(prefix="magi_"))
        try:
            config_path = self._build_config(workdir)
            raw_output = workdir / "raw_output.mp4"

            venv_torchrun = _kit_root() / "envs" / "magi-24b-v2v" / "bin" / "torchrun"
            torchrun = str(venv_torchrun) if venv_torchrun.exists() else (shutil.which("torchrun") or "torchrun")
            # --standalone picks a free rendezvous port, so two 4-GPU MAGI
            # groups on one host don't collide on a fixed port
            cmd = [
                torchrun, "--standalone",
                "--nnodes=1", f"--nproc_per_node={self.num_gpus}",
                "inference/pipeline/entry.py",
                "--config_file", str(config_path),
                "--mode", "v2v",
                "--prompt", text_prompt,
                "--prefix_video_path", str(video_path),
                "--output_path", str(raw_output),
            ]

            env = dict(os.environ)
            env["PYTHONPATH"] = f"{self.repo_path}:{env.get('PYTHONPATH', '')}"
            env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
            env.setdefault("NCCL_ALGO", "^NVLS")
            env.setdefault("PAD_HQ", "1")
            env.setdefault("PAD_DURATION", "1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("OFFLOAD_T5_CACHE", "true")
            env.setdefault("OFFLOAD_VAE_CACHE", "true")
            # include 8.0 so A100 kernels compile (issue #22); 8.9/9.0 upstream default
            env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0")

            proc = subprocess.run(
                cmd, cwd=str(self.repo_path), env=env,
                capture_output=True, text=True, timeout=14400,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"MAGI entry.py failed (exit {proc.returncode}): {proc.stderr[-800:]}")
            if not raw_output.exists():
                raise RuntimeError(f"MAGI exited 0 but produced no video. stdout tail: {proc.stdout[-400:]}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if keep_prefix:
                shutil.move(str(raw_output), str(output_path))
            else:
                self._trim_prefix(raw_output, output_path)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return {
            "video_path": str(output_path),
            "duration_seconds": time.time() - start_time,
            "status": "success",
            "metadata": {
                "mode": "continuation (prefix v2v)",
                "prefix_frames_used": _PREFIX_FRAMES,
                "prefix_trimmed": not keep_prefix,
                "num_gpus": self.num_gpus,
            },
        }


class MagiWrapper(ModelWrapper):
    """v2v-inferkit wrapper for MAGI-1 24B base (--mode v2v continuation)."""

    def __init__(
        self,
        model: str = "sand-ai/MAGI-1:24B_base",
        output_dir: str = "./outputs",
        **kwargs,
    ):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = MagiService(model=model)

    def generate(
        self,
        image_path: Optional[Union[str, Path]],
        text_prompt: str,
        duration: float = 4.0,
        output_filename: Optional[str] = None,
        video_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        if video_path is None:
            return {
                "success": False,
                "video_path": None,
                "error": "magi-24b-v2v requires video_path (prefix video)",
                "duration_seconds": 0.0,
                "generation_id": None,
                "model": self.model,
                "status": "failed",
                "metadata": {"prompt": text_prompt},
            }

        kwargs.pop("question_data", None)
        kwargs.pop("num_frames", None)
        output_path = self.output_dir / (output_filename or "video.mp4")

        try:
            result = self.service.generate_video(
                video_path=video_path,
                text_prompt=text_prompt,
                output_path=output_path,
                **kwargs,
            )
        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "generation_id": None,
                "model": self.model,
                "status": "failed",
                "metadata": {
                    "prompt": text_prompt,
                    "modality": "v2v (continuation)",
                    "video_path_input": str(video_path),
                },
            }

        return {
            "success": True,
            "video_path": result["video_path"],
            "error": None,
            "duration_seconds": result["duration_seconds"],
            "generation_id": f"magi_{int(start_time)}",
            "model": self.model,
            "status": "success",
            "metadata": {
                "prompt": text_prompt,
                "modality": "v2v (continuation)",
                "video_path_input": str(video_path),
                **result["metadata"],
            },
        }
