"""Shared fal.ai adapter for hosted video-to-video endpoints.

The model catalog supplies an endpoint and a small payload profile.  This
keeps upload, queue polling, result normalization, and downloading identical
across providers while preserving their different input field names.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import httpx

from .base import ModelWrapper

logger = logging.getLogger(__name__)


_PROFILES: Dict[str, Dict[str, Any]] = {
    "luma_ray_3_2": {
        "video_field": "video_url",
        "duration_values": {5, 10},
        "duration_type": "seconds_string",
        "allowed_controls": {
            "resolution", "duration", "auto_controls", "hdr", "exr_export",
        },
        "defaults": {
            "resolution": "720p",
            "duration": "5s",
            "auto_controls": True,
        },
    },
    "kling_o1_edit": {
        "video_field": "video_url",
        "input_min": 3.0,
        "input_max": 10.0,
        "allowed_controls": {"keep_audio"},
        "defaults": {"keep_audio": True},
    },
    "wan27_edit": {
        "video_field": "video_url",
        "input_min": 2.0,
        "input_max": 10.0,
        "duration_values": set(range(2, 11)),
        "duration_type": "string",
        "allowed_controls": {
            "resolution", "duration", "aspect_ratio", "audio_setting", "seed",
            "enable_safety_checker",
        },
        "defaults": {
            "resolution": "1080p",
            "duration": "0",
            "audio_setting": "auto",
            "enable_safety_checker": True,
        },
    },
    "gemini_omni": {
        "video_field": "video_url",
    },
    "wan3": {
        "video_field": "reference_video_urls",
        "video_is_list": True,
        "reference_label": "Video 1",
        "input_max": 15.0,
        "output_min": 2,
        "output_max": 30,
        "duration_type": "integer",
        "allowed_controls": {
            "resolution", "duration", "aspect_ratio", "seed", "audio",
            "enable_prompt_expansion", "enable_thinking", "enable_safety_checker",
        },
        "defaults": {
            "resolution": "720p",
            "aspect_ratio": "adaptive",
            "audio": True,
            "enable_prompt_expansion": True,
        },
    },
    "minimax_h3": {
        "video_field": "reference_video_urls",
        "video_is_list": True,
        "reference_label": "Video 1",
        "input_min": 2.0,
        "input_max": 15.0,
        "output_min": 4,
        "output_max": 15,
        "duration_type": "integer",
        "allowed_controls": {"resolution", "duration", "aspect_ratio", "seed"},
        "defaults": {"resolution": "768P", "aspect_ratio": "adaptive"},
    },
    "seedance_2": {
        "video_field": "video_urls",
        "video_is_list": True,
        "reference_label": "@Video1",
        "input_min": 2.0,
        "input_max": 15.0,
        "output_min": 4,
        "output_max": 15,
        "duration_type": "string",
        "allowed_controls": {
            "resolution", "duration", "aspect_ratio", "seed", "generate_audio", "end_user_id",
        },
        "defaults": {
            "resolution": "720p",
            "aspect_ratio": "auto",
            "generate_audio": True,
        },
    },
    "seedance_2_5": {
        "video_field": "video_urls",
        "video_is_list": True,
        "reference_label": "[Video1]",
        "input_max": 30.0,
        "output_min": 4,
        "output_max": 30,
        "duration_type": "string",
        "allowed_controls": {
            "resolution", "duration", "aspect_ratio", "seed", "generate_audio", "end_user_id",
        },
        "defaults": {
            "resolution": "720p",
            "aspect_ratio": "auto",
            "generate_audio": True,
        },
    },
    "gemini_omni_edit": {
        "video_field": "video_url",
        "allowed_controls": {"resolution"},
        "defaults": {"resolution": "720p"},
    },
    "kling_o3_edit": {
        "video_field": "video_url",
        "reference_label": "@Video1",
        "input_min": 3.0,
        "input_max": 15.0,
        "allowed_controls": {"keep_audio", "shot_type"},
        "defaults": {"keep_audio": True, "shot_type": "customize"},
    },
    "happy_horse_edit": {
        "video_field": "video_url",
        "input_min": 3.0,
        # The API accepts longer inputs, but edit output is capped at 15 s.
        # Rejecting longer clips avoids silently benchmarking a partial edit.
        "input_max": 15.0,
        "allowed_controls": {"resolution", "audio_setting", "seed", "enable_safety_checker"},
        "defaults": {"resolution": "720p", "audio_setting": "origin"},
    },
    "grok_edit": {
        "video_field": "video_url",
        # fal truncates longer inputs. Reject them before a paid request.
        "input_max": 8.0,
        "allowed_controls": {"resolution"},
        "defaults": {"resolution": "720p"},
    },
}

_CONTROL_KEYS = {
    "resolution",
    "duration",
    "aspect_ratio",
    "seed",
    "audio",
    "generate_audio",
    "keep_audio",
    "audio_setting",
    "shot_type",
    "enable_prompt_expansion",
    "enable_thinking",
    "enable_safety_checker",
    "end_user_id",
    "auto_controls",
    "hdr",
    "exr_export",
}


class FalRequestTimeout(TimeoutError):
    """A local wait expired while the paid remote fal request may continue."""

    def __init__(self, request_id: str, timeout: int) -> None:
        self.request_id = request_id
        super().__init__(
            f"fal request {request_id} did not finish within {timeout}s; "
            "the remote request may still be running"
        )


def _probe_video_duration(video_path: Union[str, Path]) -> float:
    """Return media duration in seconds using ffprobe."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to validate fal V2V inputs")

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip() or 'unknown error'}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse input video duration: {result.stdout!r}") from exc
    if duration <= 0:
        raise ValueError("Input video has no positive duration")
    return duration


def _has_audio_stream(video_path: Union[str, Path]) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _pad_video(video_path: Path, source_duration: float, target_duration: float) -> Path:
    """Extend a short clip by cloning its final frame (and padding audio)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to pad a too-short fal V2V input")

    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    handle.close()
    output_path = Path(handle.name)
    padding = max(0.0, target_duration - source_duration)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-vf", f"tpad=stop_mode=clone:stop_duration={padding:.3f}"]
    if _has_audio_stream(video_path):
        cmd += ["-af", f"apad=pad_dur={padding:.3f}", "-c:a", "aac"]
    cmd += [
        "-t",
        f"{target_duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg padding failed: {result.stderr[-500:]}")
    return output_path


class FalV2VService:
    """Upload, queue, poll, and download one fal V2V generation."""

    def __init__(
        self,
        endpoint: str,
        profile: str,
        max_wait: int = 1800,
        poll_interval: float = 5.0,
        **_: Any,
    ) -> None:
        if profile not in _PROFILES:
            raise ValueError(f"Unknown fal V2V profile: {profile}")
        self.endpoint = endpoint
        self.profile_name = profile
        self.profile = _PROFILES[profile]
        self.max_wait = max_wait
        self.poll_interval = poll_interval

    @staticmethod
    def _fal_client():
        if not os.environ.get("FAL_KEY"):
            raise ValueError("FAL_KEY not found. Set it in your environment or .env file.")
        try:
            import fal_client
        except ImportError as exc:
            raise RuntimeError("fal-client is not installed; run `pip install -e .`") from exc
        return fal_client

    def prepare_input(self, video_path: Union[str, Path]) -> Tuple[Path, float, bool]:
        """Validate and optionally pad an input; return path, duration, is_temp."""
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Conditioning video not found: {path}")

        duration = _probe_video_duration(path)
        maximum = self.profile.get("input_max")
        if maximum is not None and duration > float(maximum) + 0.01:
            raise ValueError(
                f"{self.endpoint} accepts at most {maximum:g}s of source video; "
                f"input is {duration:.3f}s"
            )

        minimum = self.profile.get("input_min")
        if minimum is not None and duration < float(minimum):
            padded = _pad_video(path, duration, float(minimum))
            return padded, float(minimum), True
        return path, duration, False

    def build_payload(
        self,
        prompt: str,
        video_url: str,
        source_duration: float,
        requested_duration: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Translate common runner inputs to the selected endpoint schema."""
        profile = self.profile
        reference_label = profile.get("reference_label")
        rendered_prompt = prompt
        if reference_label and reference_label.lower() not in prompt.lower():
            rendered_prompt = f"Apply this edit to {reference_label}: {prompt}"

        payload: Dict[str, Any] = {"prompt": rendered_prompt}
        video_value: Any = [video_url] if profile.get("video_is_list") else video_url
        payload[profile["video_field"]] = video_value
        payload.update(profile.get("defaults", {}))

        if "output_min" in profile:
            minimum = int(profile["output_min"])
            maximum = int(profile["output_max"])
            if requested_duration is not None:
                if requested_duration < minimum or requested_duration > maximum:
                    raise ValueError(
                        f"Requested duration must be between {minimum} and {maximum} seconds "
                        f"for {self.endpoint}"
                    )
                output_duration = int(math.ceil(requested_duration))
            else:
                output_duration = max(minimum, min(maximum, int(math.ceil(source_duration))))
            payload["duration"] = (
                str(output_duration)
                if profile.get("duration_type") == "string"
                else output_duration
            )
        elif requested_duration is not None and "duration_values" in profile:
            output_duration = int(math.ceil(requested_duration))
            if output_duration not in profile["duration_values"]:
                choices = ", ".join(str(value) for value in sorted(profile["duration_values"]))
                raise ValueError(
                    f"Requested duration must be one of {choices} seconds for {self.endpoint}"
                )
            duration_type = profile.get("duration_type")
            if duration_type == "seconds_string":
                payload["duration"] = f"{output_duration}s"
            elif duration_type == "string":
                payload["duration"] = str(output_duration)
            else:
                payload["duration"] = output_duration

        allowed_controls = profile.get("allowed_controls", set())
        for key, value in kwargs.items():
            if key in allowed_controls and key != "duration" and value is not None:
                payload[key] = value
        return payload

    def submit(self, payload: Dict[str, Any], max_wait: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        fal_client = self._fal_client()
        handler = fal_client.submit(self.endpoint, arguments=payload)
        request_id = handler.request_id
        timeout = self.max_wait if max_wait is None else max_wait
        started = time.monotonic()

        while time.monotonic() - started < timeout:
            status = fal_client.status(self.endpoint, request_id, with_logs=True)
            status_name = status.__class__.__name__.lower()
            if "completed" in status_name:
                return fal_client.result(self.endpoint, request_id), request_id
            if "failed" in status_name or "cancel" in status_name:
                detail = getattr(status, "error", None) or getattr(status, "detail", None) or status
                raise RuntimeError(f"fal request {request_id} failed: {detail}")
            if self.poll_interval:
                time.sleep(self.poll_interval)

        raise FalRequestTimeout(request_id, timeout)

    @staticmethod
    def download(video_url: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            with client.stream("GET", video_url) as response:
                response.raise_for_status()
                with output_path.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
        return output_path

    def generate_video(
        self,
        prompt: str,
        video_path: Union[str, Path],
        output_path: Path,
        duration: Optional[float] = None,
        max_wait: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        fal_client = self._fal_client()
        prepared_path, input_duration, is_temp = self.prepare_input(video_path)
        try:
            video_url = fal_client.upload_file(str(prepared_path))
            if not video_url:
                raise RuntimeError("fal returned no URL for the uploaded source video")
            payload = self.build_payload(
                prompt,
                video_url,
                input_duration,
                requested_duration=duration,
                **kwargs,
            )
            response, request_id = self.submit(payload, max_wait=max_wait)
            result_video = response.get("video") or {}
            result_url = result_video.get("url")
            if not result_url:
                raise RuntimeError(f"fal request {request_id} completed without a video URL")
            self.download(result_url, output_path)
            return {
                "video_path": str(output_path),
                "video_url": result_url,
                "request_id": request_id,
                "endpoint": self.endpoint,
                "payload": payload,
                "input_duration": input_duration,
                "response": response,
            }
        finally:
            if is_temp:
                prepared_path.unlink(missing_ok=True)


class FalV2VWrapper(ModelWrapper):
    """Standard eight-field wrapper shared by hosted fal V2V models."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        profile: str,
        output_dir: str = "./outputs",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = FalV2VService(endpoint=endpoint, profile=profile, **kwargs)

    def generate(
        self,
        image_path: Optional[Union[str, Path]] = None,
        text_prompt: str = "",
        duration: Optional[float] = None,
        output_filename: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        started = time.time()
        video_path = kwargs.pop("video_path", None)
        output_path = Path(self.output_dir) / (output_filename or "video.mp4")

        try:
            if not video_path:
                raise ValueError("video_path is required for fal V2V inference")
            allowed_kwargs = {key: value for key, value in kwargs.items() if key in _CONTROL_KEYS}
            max_wait = kwargs.get("max_wait")
            result = self.service.generate_video(
                prompt=text_prompt,
                video_path=video_path,
                output_path=output_path,
                duration=duration,
                max_wait=max_wait,
                **allowed_kwargs,
            )
            response = result["response"]
            return {
                "success": True,
                "video_path": result["video_path"],
                "error": None,
                "duration_seconds": time.time() - started,
                "generation_id": result["request_id"],
                "model": self.model,
                "status": "success",
                "metadata": {
                    "provider": "fal",
                    "endpoint": result["endpoint"],
                    "video_url": result["video_url"],
                    "input_duration": result["input_duration"],
                    "requested_duration": result["payload"].get("duration"),
                    "resolution": result["payload"].get("resolution"),
                    "seed": response.get("seed"),
                    "interaction_id": response.get("interaction_id"),
                    "actual_prompt": response.get("actual_prompt"),
                },
            }
        except Exception as exc:
            logger.error("fal V2V generation failed for %s: %s", self.model, exc)
            request_id = getattr(exc, "request_id", None)
            return {
                "success": False,
                "video_path": None,
                "error": str(exc),
                "duration_seconds": time.time() - started,
                "generation_id": request_id,
                "model": self.model,
                "status": "failed",
                "metadata": {
                    "provider": "fal",
                    "endpoint": self.service.endpoint,
                    "request_id": request_id,
                    "prompt": text_prompt,
                    "image_path": str(image_path) if image_path else None,
                },
            }
