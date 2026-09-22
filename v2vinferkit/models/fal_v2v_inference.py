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
        "duration_type": "integer",
        "allowed_controls": {
            "resolution", "duration", "aspect_ratio", "audio_setting", "seed",
            "enable_safety_checker",
        },
        "defaults": {
            "resolution": "1080p",
            "duration": 0,
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
            # Benchmark defaults: no audio (silent sources, billed extra) and no
            # server-side prompt rewriting (the prompt must reach the model as written).
            "audio": False,
            "enable_prompt_expansion": False,
        },
    },
    "minimax_h3": {
        "video_field": "reference_video_urls",
        "video_is_list": True,
        "input_min": 2.0,
        "input_max": 15.0,
        "output_min": 5,
        "output_max": 15,
        "duration_type": "integer",
        "allowed_controls": {"resolution", "duration", "aspect_ratio", "seed"},
        "defaults": {"resolution": "768P", "aspect_ratio": "adaptive"},
    },
    "seedance_2": {
        "video_field": "video_urls",
        "video_is_list": True,
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
            "generate_audio": False,
        },
    },
    "seedance_2_5": {
        "video_field": "video_urls",
        "video_is_list": True,
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
            "generate_audio": False,
        },
    },
    "gemini_omni_edit": {
        "video_field": "video_url",
        "allowed_controls": {"resolution"},
        "defaults": {"resolution": "720p"},
    },
    "kling_o3_edit": {
        "video_field": "video_url",
        "input_min": 3.0,
        "input_max": 15.0,
        "allowed_controls": {"keep_audio", "shot_type"},
        "defaults": {"keep_audio": False, "shot_type": "customize"},
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
    "grok_extend": {
        # True continuation: generates new frames after the source's last frame.
        # `duration` = extension length, integer seconds 2-10; the returned file is
        # the ORIGINAL + EXTENSION stitched together (fal schema says so), so the
        # source span must be trimmed before scoring.
        "video_field": "video_url",
        "extension": True,
        "input_min": 2.0,
        "input_max": 15.0,
        "duration_type": "integer",
        "duration_values": set(range(2, 11)),
        # resolution: fal bills 480p at $0.06/s vs 720p $0.08/s (+$0.01/s input either way)
        "allowed_controls": {"duration", "resolution"},
        "defaults": {"duration": 6},
    },
    "veo31_extend": {
        # Veo 3.1 extend: `duration` is a const "7s" and resolution a const "720p"
        # in fal's schema, so the ground-truth length cannot be honoured — the
        # output has to be trimmed afterwards. Audio is on by default upstream
        # and doubles the price; off here.
        "video_field": "video_url",
        "extension": True,
        "fixed_duration": True,
        "input_max": 8.0,
        "allowed_controls": {"seed", "negative_prompt", "generate_audio", "aspect_ratio", "resolution"},
        "defaults": {
            "duration": "7s",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "generate_audio": False,
        },
    },
    "ltx23_extend": {
        # LTX-2.3 Pro extend: `duration` is a float 2-20 (seconds of NEW content),
        # `mode` end/start, `context` = seconds of source used as context. The only
        # hosted extend endpoint that can match a 2.5 s ground truth exactly.
        "video_field": "video_url",
        # fal rejects sources under 73 frames (`video_too_few_frames`, seen
        # 2026-08-30 on a 60-frame / 2.5 s benchmark clip). 73 frames at 24 fps
        # is 3.04 s; pad to 3.25 s (78 frames) so frame rounding can never land
        # short. Padding clones the FIRST frame, so the continuation seam is intact.
        "input_min": 3.25,
        "extension": True,
        "duration_type": "float",
        "duration_float_min": 2.0,
        "duration_float_max": 20.0,
        "context_from_source": True,
        "allowed_controls": {"duration", "mode", "context"},
        "defaults": {"mode": "end"},
    },
}

_CONTROL_KEYS = {
    "resolution",
    "duration",
    "negative_prompt",
    "mode",
    "context",
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
    """A local wait expired; the remote request was asked to cancel."""

    def __init__(self, request_id: str, timeout: int, cancelled: bool = False) -> None:
        self.request_id = request_id
        self.cancelled = cancelled
        super().__init__(
            f"fal request {request_id} did not finish within {timeout}s; "
            + ("cancel requested" if cancelled else "cancel FAILED - the remote request may still be running and billing")
        )


class FalRequestFailed(RuntimeError):
    """A submitted (billed) fal request ended without a usable result.

    Carries the request id so the failure can be traced in the fal dashboard
    even when the SDK exception itself has no id (FalClientHTTPError does not).
    """

    def __init__(self, request_id: Optional[str], message: str) -> None:
        self.request_id = request_id
        super().__init__(f"fal request {request_id or '<not submitted>'} failed: {message}")


def _probe_video_geometry(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Return width/height/fps/frames/duration of the first video stream (ffprobe)."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to validate generated videos")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of", "json", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip() or 'unknown error'}")
    import json as _json
    data = _json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("no video stream")
    st = streams[0]
    num, _, den = str(st.get("avg_frame_rate", "0/1")).partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    frames = int(st.get("nb_read_frames") or 0)
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if frames <= 0 or duration <= 0:
        raise ValueError(f"video has no decodable frames (frames={frames}, duration={duration})")
    return {
        "width": int(st.get("width", 0)),
        "height": int(st.get("height", 0)),
        "fps": round(fps, 3),
        "frames": frames,
        "duration": round(duration, 3),
    }


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


def _probe_video_fps(video_path: Union[str, Path]) -> float:
    """Return the average frame rate of the first video stream using ffprobe."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to derive an extension length")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip() or 'unknown error'}")
    raw = result.stdout.strip()
    try:
        numerator, _, denominator = raw.partition("/")
        fps = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Could not parse input video frame rate: {raw!r}") from exc
    if fps <= 0:
        raise ValueError("Input video has no positive frame rate")
    return fps


def extension_seconds(video_path: Union[str, Path], num_frames: int, integer: bool = True) -> float:
    """Seconds an extension endpoint must generate to cover `num_frames`.

    Benchmark tasks ship a ground-truth continuation; its frame count at the
    source's frame rate is the length the model has to produce. Integer
    endpoints get the value rounded UP so the answer is never shorter than the
    reference; float endpoints get the exact value.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive to derive an extension length")
    exact = num_frames / _probe_video_fps(video_path)
    if integer:
        return max(1, int(math.ceil(exact)))
    return round(exact, 3)


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
    """Extend a short clip by cloning its FIRST frame in front of it.

    The padding goes at the start, never the end: a benchmark clip ends at the
    moment a continuation has to pick up, and cloning the last frame there
    would show the model a frozen scene exactly where the event goes on. The
    opening frames are an establishing shot, so a longer hold there changes
    nothing. Audio (if any) is delayed by the same amount.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to pad a too-short fal V2V input")

    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    handle.close()
    output_path = Path(handle.name)
    padding = max(0.0, target_duration - source_duration)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-vf", f"tpad=start_mode=clone:start_duration={padding:.3f}"]
    if _has_audio_stream(video_path):
        cmd += ["-af", f"adelay={int(round(padding * 1000))}:all=1", "-c:a", "aac"]
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
        self.last_request_id: Optional[str] = None
        self.last_payload: Optional[Dict[str, Any]] = None

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
        # The prompt is sent verbatim. No provider-specific reference labels or
        # "apply this edit" framing: a benchmark prompt describes an event, and
        # rewriting it changes what is being measured.
        payload: Dict[str, Any] = {"prompt": prompt}
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
        elif requested_duration is not None and profile.get("duration_type") == "integer":
            # Free integer seconds (extension endpoints publish no range).
            payload["duration"] = int(math.ceil(requested_duration))
        elif requested_duration is not None and profile.get("duration_type") == "float":
            lo = float(profile.get("duration_float_min", 0))
            hi = float(profile.get("duration_float_max", float("inf")))
            if requested_duration < lo or requested_duration > hi:
                raise ValueError(
                    f"Requested duration must be between {lo:g} and {hi:g} seconds for {self.endpoint}"
                )
            payload["duration"] = round(float(requested_duration), 3)
        elif requested_duration is not None and profile.get("fixed_duration"):
            raise ValueError(
                f"{self.endpoint} has a fixed output duration ({profile['defaults'].get('duration')}); "
                "a requested duration cannot be honoured"
            )

        if profile.get("context_from_source"):
            # Give the extend endpoint the whole source as context (capped by its schema).
            payload["context"] = round(min(float(source_duration), 20.0), 3)

        allowed_controls = profile.get("allowed_controls", set())
        for key, value in kwargs.items():
            if key in allowed_controls and key != "duration" and value is not None:
                payload[key] = value
        return payload

    def submit(self, payload: Dict[str, Any], max_wait: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        """Submit one paid request and poll it to completion.

        Every exception raised after submission carries the request id
        (FalRequestFailed / FalRequestTimeout) — the id is the only handle on
        a billed job, and fal's own HTTP errors do not include it.
        """
        fal_client = self._fal_client()
        handler = fal_client.submit(self.endpoint, arguments=payload)
        request_id = handler.request_id
        self.last_request_id = request_id
        timeout = self.max_wait if max_wait is None else max_wait
        started = time.monotonic()

        while True:
            status = fal_client.status(self.endpoint, request_id, with_logs=True)
            status_name = status.__class__.__name__.lower()
            if "completed" in status_name:
                # fal_client has no Failed status class: a failed job arrives as
                # Completed(error=..., error_type=...).
                error = getattr(status, "error", None)
                if error:
                    raise FalRequestFailed(request_id, f"{getattr(status, 'error_type', '') or ''} {error}".strip())
                try:
                    return fal_client.result(self.endpoint, request_id), request_id
                except Exception as exc:  # FalClientHTTPError carries no request id
                    raise FalRequestFailed(request_id, f"{exc.__class__.__name__}: {exc}") from exc
            if "failed" in status_name or "cancel" in status_name:
                detail = getattr(status, "error", None) or getattr(status, "detail", None) or status
                raise FalRequestFailed(request_id, str(detail))
            if time.monotonic() - started >= timeout:
                cancelled = False
                try:
                    fal_client.cancel(self.endpoint, request_id)
                    cancelled = True
                except Exception as exc:
                    logger.error("fal cancel failed for %s: %s", request_id, exc)
                raise FalRequestTimeout(request_id, timeout, cancelled=cancelled)
            if self.poll_interval:
                time.sleep(self.poll_interval)

    @staticmethod
    def download(video_url: str, output_path: Path) -> Path:
        """Stream to `<name>.part`, verify, then rename — a broken download never
        leaves a file at the final path (which skip-existing would trust)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = output_path.with_name(output_path.name + ".part")
        try:
            with httpx.Client(timeout=600.0, follow_redirects=True) as client:
                with client.stream("GET", video_url) as response:
                    response.raise_for_status()
                    with part_path.open("wb") as output:
                        for chunk in response.iter_bytes():
                            output.write(chunk)
            if part_path.stat().st_size == 0:
                raise RuntimeError("downloaded video is empty (0 bytes)")
            part_path.replace(output_path)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise
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
        self.last_request_id = None
        source_duration = _probe_video_duration(video_path)
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
            self.last_payload = payload
            response, request_id = self.submit(payload, max_wait=max_wait)
            result_video = response.get("video") or {}
            result_url = result_video.get("url")
            if not result_url:
                raise FalRequestFailed(request_id, "completed without a video URL")
            self.download(result_url, output_path)
            try:
                output_geometry = _probe_video_geometry(output_path)
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                raise FalRequestFailed(request_id, f"downloaded video failed validation: {exc}") from exc
            return {
                "video_path": str(output_path),
                "video_url": result_url,
                "request_id": request_id,
                "endpoint": self.endpoint,
                "profile": self.profile_name,
                "payload": payload,
                "prompt_sent": payload.get("prompt"),
                "source_duration": source_duration,
                "input_duration": input_duration,
                "input_padded_seconds": round(input_duration - source_duration, 3) if is_temp else 0.0,
                "output_geometry": output_geometry,
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
        num_frames = kwargs.pop("num_frames", None)
        output_path = Path(self.output_dir) / (output_filename or "video.mp4")

        try:
            if not video_path:
                raise ValueError("video_path is required for fal V2V inference")
            profile = self.service.profile
            if profile.get("extension") and not profile.get("fixed_duration") and duration is None:
                # Continuation endpoints: generate exactly as long as the ground
                # truth. Never fall back to the provider default — that silently
                # changes what is billed (grok defaults to 6 s, twice the need).
                if not num_frames:
                    raise ValueError(
                        f"{self.model}: extension length unknown — the task has no readable "
                        "ground_truth.mp4 and no --duration was given"
                    )
                duration = extension_seconds(
                    video_path, int(num_frames), integer=profile.get("duration_type") != "float"
                )
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
                    "profile": result.get("profile", self.service.profile_name),
                    "request_id": result["request_id"],
                    "video_url": result["video_url"],
                    "payload": dict(result.get("payload") or {}),
                    "prompt_sent": result.get("prompt_sent", (result.get("payload") or {}).get("prompt")),
                    "prompt_original": text_prompt,
                    "source_duration": result.get("source_duration"),
                    "input_duration": result.get("input_duration"),
                    "input_padded_seconds": result.get("input_padded_seconds", 0.0),
                    "requested_duration": (result.get("payload") or {}).get("duration"),
                    "resolution": (result.get("payload") or {}).get("resolution"),
                    "seed": response.get("seed", (result.get("payload") or {}).get("seed")),
                    "output_geometry": result.get("output_geometry"),
                    "interaction_id": response.get("interaction_id"),
                    "actual_prompt": response.get("actual_prompt"),
                },
            }
        except Exception as exc:
            logger.error("fal V2V generation failed for %s: %s", self.model, exc)
            request_id = getattr(exc, "request_id", None) or getattr(self.service, "last_request_id", None)
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
                    "profile": self.service.profile_name,
                    "request_id": request_id,
                    "cancelled": getattr(exc, "cancelled", None),
                    "payload": getattr(self.service, "last_payload", None),
                    "prompt_original": text_prompt,
                    "image_path": str(image_path) if image_path else None,
                },
            }
