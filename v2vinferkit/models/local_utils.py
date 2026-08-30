"""Shared helpers for local/open-weight model integrations."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union


KIT_ROOT = Path(__file__).resolve().parents[2]


class LocalInferenceError(RuntimeError):
    """Raised when an upstream local inference command cannot produce output."""


def weights_path(*parts: str) -> Path:
    root = Path(os.environ.get("V2V_WEIGHTS_DIR") or str(KIT_ROOT / "weights"))
    return root.joinpath(*parts)


def repo_path(name: str, env_name: Optional[str] = None) -> Path:
    if env_name and os.environ.get(env_name):
        return Path(os.environ[env_name]).expanduser().resolve()
    return KIT_ROOT / "repos" / name


def require_file(path: Union[str, Path], label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LocalInferenceError(f"{label} not found: {resolved}")
    return resolved


def require_dir(path: Union[str, Path], label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise LocalInferenceError(f"{label} not found: {resolved}")
    return resolved


def run_command(
    cmd: Sequence[Union[str, Path]],
    *,
    cwd: Union[str, Path],
    timeout: int = 7200,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run an upstream CLI without a shell and retain useful failure context."""
    command = [str(part) for part in cmd]
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalInferenceError(
            f"Inference command timed out after {timeout}s: {' '.join(command[:3])}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no process output")[-1600:]
        raise LocalInferenceError(
            f"Inference command failed (exit {result.returncode}): {detail}"
        )
    return result


def newest_video(directory: Union[str, Path], patterns: Iterable[str] = ("*.mp4",)) -> Path:
    base = Path(directory)
    candidates = []
    for pattern in patterns:
        candidates.extend(base.rglob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise LocalInferenceError(f"Inference exited successfully but produced no video under {base}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def failed_result(
    model: str,
    prompt: str,
    start_time: float,
    error: Union[str, Exception],
    video_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    return {
        "success": False,
        "video_path": None,
        "error": str(error),
        "duration_seconds": time.time() - start_time,
        "generation_id": None,
        "model": model,
        "status": "failed",
        "metadata": {
            "prompt": prompt,
            "modality": "v2v",
            "video_path_input": str(video_path) if video_path is not None else None,
        },
    }


def success_result(
    model: str,
    prompt: str,
    start_time: float,
    output_path: Union[str, Path],
    input_video: Union[str, Path],
    generation_prefix: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "success": True,
        "video_path": str(output_path),
        "error": None,
        "duration_seconds": time.time() - start_time,
        "generation_id": f"{generation_prefix}_{int(start_time)}",
        "model": model,
        "status": "success",
        "metadata": {
            "prompt": prompt,
            "modality": "v2v",
            "video_path_input": str(input_video),
            **(metadata or {}),
        },
    }
