"""
MiniMax H3 (open weights, Ref2VA) served locally by SGLang.

The model card's local deployment is `sglang serve --model-path MiniMaxAI/MiniMax-H3
--model-variant ref2va` (4 GPUs, ulysses 4); this wrapper is an HTTP client to that
server's /v1/videos endpoint, so it runs in the kit's own interpreter (no venv).
Conditioning: the task's input video is the `reference` video; a reference image
(first_frame.png, when the task has one) is added as an image reference. Output
duration follows ground_truth.mp4 (clamped to the 5–15 s the model accepts);
sources shorter than 2 s are front-padded by cloning the first frame, sources
longer than 15 s are cut to their first 15 s — both recorded in the result.
The audio track is stripped from the returned mp4 (benchmark compares video only).

Env: MINIMAX_H3_URL (default http://127.0.0.1:30011), MINIMAX_H3_SHORT_EDGE (768),
MINIMAX_H3_TIMEOUT seconds per generation (3600).
"""
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper

INPUT_MIN_S, INPUT_MAX_S = 2.0, 15.0
OUTPUT_MIN_S, OUTPUT_MAX_S = 5, 15


def _probe(path: Union[str, Path]) -> Dict[str, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    n = int(s.get("nb_frames") or 0)
    dur = float((d.get("format") or {}).get("duration") or (n / fps if fps else 0))
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "frames": n, "duration": dur}


def _http(method: str, url: str, body: Optional[dict] = None, timeout: int = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw


class MiniMaxH3LocalWrapper(ModelWrapper):
    def __init__(self, model: str = "MiniMaxAI/MiniMax-H3:ref2va", output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.base_url = os.environ.get("MINIMAX_H3_URL", "http://127.0.0.1:30011").rstrip("/")
        self.short_edge = int(os.environ.get("MINIMAX_H3_SHORT_EDGE", "768"))
        self.timeout = int(os.environ.get("MINIMAX_H3_TIMEOUT", "3600"))

    def _prepare_source(self, video_path: Path, workdir: Path) -> Dict[str, Any]:
        info = _probe(video_path)
        src, pad, cut = video_path, 0.0, 0.0
        if info["duration"] < INPUT_MIN_S:
            pad = INPUT_MIN_S + 0.1 - info["duration"]
            src = workdir / "padded.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
                            "-vf", f"tpad=start_duration={pad:.3f}:start_mode=clone",
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(src)],
                           capture_output=True, check=True)
        elif info["duration"] > INPUT_MAX_S:
            cut = info["duration"] - INPUT_MAX_S
            src = workdir / "cut.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-t", str(INPUT_MAX_S),
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(src)],
                           capture_output=True, check=True)
        return {"path": src, "info": info, "input_padded_seconds": round(pad, 3), "input_cut_seconds": round(cut, 3)}

    def generate(
        self,
        image_path: Optional[Union[str, Path]],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        video_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start = time.time()
        fail = lambda err, meta=None: {  # noqa: E731
            "success": False, "video_path": None, "error": err, "duration_seconds": time.time() - start,
            "generation_id": None, "model": self.model, "status": "failed",
            "metadata": {"prompt": text_prompt, **(meta or {})},
        }
        if video_path is None:
            return fail("minimax-h3-local is video-conditioned: video_path is required")
        qd = kwargs.get("question_data") or {}
        seed = int(kwargs.get("seed", 0) or 0)
        output_path = self.output_dir / (output_filename or "video.mp4")
        workdir = Path(tempfile.mkdtemp(prefix="h3_"))
        try:
            src = self._prepare_source(Path(video_path), workdir)
            # output length follows the target clip when the task has one
            gt = qd.get("ground_truth_video")
            target_s = _probe(gt)["duration"] if gt and Path(gt).exists() else src["info"]["duration"]
            out_s = max(OUTPUT_MIN_S, min(OUTPUT_MAX_S, int(math.ceil(target_s))))
            conditions = [{"type": "video", "uri": f"file://{Path(src['path']).resolve()}", "role": "reference"}]
            if image_path and Path(image_path).exists():
                conditions.append({"type": "image", "uri": f"file://{Path(image_path).resolve()}", "role": "reference"})
            # Ref2VA prompts name their references (<Video 1>, <Picture 1>…); the bench prompt
            # is passed as written, with one line binding the input clip to <Video 1>.
            refs_line = "<Video 1> is the source video for this task; the target video is the result of applying the instruction above to <Video 1>."
            if len(conditions) > 1:
                refs_line += " <Picture 1> is a reference image for the task."
            full_prompt = f"{text_prompt.strip()}\n\n{refs_line}"
            body = {
                "task": "ref2va",
                "prompt": full_prompt,
                "conditions": conditions,
                "target": {"short_edge": self.short_edge, "aspect_ratio": "auto", "duration_seconds": out_s},
                "seed": seed,
            }
            resp = json.loads(_http("POST", f"{self.base_url}/v1/videos", body))
            vid = resp.get("id")
            if not vid:
                return fail(f"no id in response: {str(resp)[:300]}", {"request": body})
            status = None
            deadline = start + self.timeout
            while time.time() < deadline:
                st = json.loads(_http("GET", f"{self.base_url}/v1/videos/{vid}"))
                status = st.get("status")
                if status in ("completed", "succeeded", "done"):
                    break
                if status in ("failed", "error", "cancelled"):
                    return fail(f"server status {status}: {str(st)[:400]}", {"request": body, "video_id": vid})
                time.sleep(5)
            else:
                return fail(f"timeout after {self.timeout}s (status {status})", {"request": body, "video_id": vid})
            raw = workdir / "raw.mp4"
            raw.write_bytes(_http("GET", f"{self.base_url}/v1/videos/{vid}/content", timeout=600))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "copy", "-an", str(output_path)],
                           capture_output=True, check=True)
            out_info = _probe(output_path)
            return {
                "success": True, "video_path": str(output_path), "error": None,
                "duration_seconds": time.time() - start, "generation_id": vid, "model": self.model,
                "status": "success",
                "metadata": {
                    "prompt": text_prompt, "modality": "v2v", "pipeline": "sglang /v1/videos ref2va (local weights)",
                    "request": {k: v for k, v in body.items() if k != "prompt"}, "prompt_sent": full_prompt,
                    "input_padded_seconds": src["input_padded_seconds"], "input_cut_seconds": src["input_cut_seconds"],
                    "input_duration_seconds": round(src["info"]["duration"], 3),
                    "output_geometry": f"{out_info['width']}x{out_info['height']}@{out_info['fps']:.3f}fps/{out_info['frames']}f",
                    "audio_stripped": True, "seed": seed,
                },
            }
        except Exception as e:  # noqa: BLE001
            return fail(str(e)[:800])
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
