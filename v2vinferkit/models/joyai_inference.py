"""JoyAI-Video-Edit adapter for its official streaming WebSocket server."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, success_result, weights_path


def _service_urls(value: str) -> Tuple[str, str]:
    value = value.rstrip("/")
    if value.startswith("ws://") or value.startswith("wss://"):
        ws_url = value if value.endswith("/ws") else f"{value}/ws"
        http_url = ("https://" if value.startswith("wss://") else "http://") + value.split("://", 1)[1]
        if http_url.endswith("/ws"):
            http_url = http_url[:-3]
        return ws_url, http_url
    http_url = value
    ws_url = ("wss://" if value.startswith("https://") else "ws://") + value.split("://", 1)[-1]
    return f"{ws_url}/ws", http_url


class JoyAIService:
    def __init__(self, model: str = "jd-opensource/JoyAI-Video-Edit"):
        self.model = model
        self.repo = repo_path("JoyAI-Video-Edit", "JOYAI_REPO_PATH")
        self.weights = Path(os.environ.get("JOYAI_WEIGHTS_PATH") or str(weights_path("JoyAI-Video-Edit")))
        self.text_encoder = Path(os.environ.get("JOYAI_TEXT_ENCODER_PATH") or str(weights_path("MiMo-VL-7B-RL-2508")))

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)

    @staticmethod
    def _wait_healthy(http_url: str, timeout: int, process: Optional[subprocess.Popen] = None) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        last_error = "server not reachable"
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"JoyAI server exited during startup with code {process.returncode}")
            try:
                with urllib.request.urlopen(f"{http_url}/health", timeout=5) as response:
                    health = json.loads(response.read().decode("utf-8"))
                if health.get("ok") and health.get("runtime_loaded"):
                    return health
            except Exception as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(f"JoyAI server was not ready after {timeout}s: {last_error}")

    def _start_server(self, work_dir: Path, startup_timeout: int) -> Tuple[str, str, subprocess.Popen, Any]:
        repo = require_dir(self.repo, "JoyAI repository")
        deploy = require_dir(repo / "deploy", "JoyAI deploy directory")
        checkpoint = require_dir(self.weights, "JoyAI checkpoint")
        text_encoder = require_dir(self.text_encoder, "JoyAI MiMo-VL text encoder")
        dit = require_file(checkpoint / "dit" / "joyai_video_edit_dit_0811.pth", "JoyAI DiT checkpoint")
        vae = require_dir(checkpoint / "vae", "JoyAI VAE checkpoint")
        port = self._free_port()
        env = os.environ.copy()
        env.update({
            "JOYOMNI_HOST": "127.0.0.1",
            "JOYOMNI_PORT": str(port),
            "JOYOMNI_DIT_CKPT": str(dit),
            "JOYOMNI_VAE_CKPT": str(vae),
            "JOYOMNI_TEXT_ENCODER_CKPT": str(text_encoder),
            "JOYOMNI_RECORD_DIR": str(work_dir / "recordings"),
        })
        if (Path(sys.prefix) / ".joyai_no_fp8").is_file():
            env["JOYOMNI_FP8_IMG"] = "0"
            env["JOYOMNI_FP8_TXT"] = "0"
        log_handle = open(work_dir / "server.log", "w", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", "run_server.sh", "--no-use-pe", "--no-online-gate"],
            cwd=str(deploy), env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        http_url = f"http://127.0.0.1:{port}"
        ws_url = f"ws://127.0.0.1:{port}/ws"
        try:
            self._wait_healthy(http_url, startup_timeout, process)
        except Exception:
            self._stop_process(process)
            log_handle.close()
            log_tail = (work_dir / "server.log").read_text(encoding="utf-8", errors="replace")[-1600:]
            raise RuntimeError(f"JoyAI managed server failed to start: {log_tail}")
        return ws_url, http_url, process, log_handle

    @staticmethod
    def _reference_payload(reference_image_path: Optional[Union[str, Path]]) -> Optional[str]:
        if not reference_image_path:
            return None
        path = require_file(reference_image_path, "reference image")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _video_frames(video_path: Path, width: int, height: int, max_frames: Optional[int]):
        import av
        from PIL import Image
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else 24.0
        try:
            for index, frame in enumerate(container.decode(stream)):
                if max_frames is not None and index >= max_frames:
                    break
                image = frame.to_image().convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                yield index + 1, index * 1000.0 / source_fps, buffer.getvalue()
        finally:
            container.close()

    def _stream_video(
        self,
        ws_url: str,
        http_url: str,
        video_path: Path,
        prompt: str,
        output_path: Path,
        *,
        reference_image_path=None,
        width=840,
        height=480,
        fps=24,
        num_frames=None,
        num_inference_steps=2,
        seed=42,
        inference_timeout=7200,
    ) -> Dict[str, Any]:
        import websocket
        ws = websocket.create_connection(ws_url, timeout=30, enable_multithread=True)
        ws.settimeout(max(60, inference_timeout))
        state: Dict[str, Any] = {"error": None, "frames_out": 0, "finalized": None}
        started = threading.Event()
        finalized = threading.Event()
        reader_done = threading.Event()

        def reader():
            try:
                while True:
                    message = ws.recv()
                    if message is None:
                        break
                    if isinstance(message, bytes):
                        continue
                    payload = json.loads(message)
                    message_type = payload.get("type")
                    if message_type == "started":
                        started.set()
                    elif message_type == "chunk_done":
                        state["frames_out"] = int(payload.get("frames_out", state["frames_out"]))
                    elif message_type == "recording_finalized":
                        state["finalized"] = payload
                        finalized.set()
                    elif message_type == "error":
                        state["error"] = payload.get("message", "JoyAI server error")
                        started.set()
                        finalized.set()
            except Exception as exc:
                if not reader_done.is_set():
                    state["error"] = str(exc)
                    started.set()
                    finalized.set()
            finally:
                reader_done.set()

        thread = threading.Thread(target=reader, name="joyai-ws-reader", daemon=True)
        thread.start()
        start_payload = {
            "type": "start", "source": "file", "prompt": prompt,
            "ref_image": self._reference_payload(reference_image_path),
            "width": width, "height": height, "fps": fps,
            "num_inference_steps": num_inference_steps, "seed": seed,
            "use_pe": False, "gate_enabled": False,
            "input_codec": "mjpeg", "output_codec": "mjpeg",
            "max_inflight_chunks": 0,
        }
        ws.send(json.dumps(start_payload))
        if not started.wait(timeout=60) or state["error"]:
            ws.close()
            raise RuntimeError(state["error"] or "JoyAI server did not acknowledge the session")

        sent = 0
        for seq, capture_ms, jpeg in self._video_frames(video_path, width, height, num_frames):
            ws.send(json.dumps({"type": "frame_meta", "seq": seq, "t_capture_ms": capture_ms}))
            ws.send_binary(jpeg)
            sent += 1
        if sent == 0:
            ws.close()
            raise ValueError(f"No frames decoded from {video_path}")

        ws.send(json.dumps({"type": "finalize_recording"}))
        if not finalized.wait(timeout=inference_timeout):
            ws.close()
            raise TimeoutError(f"JoyAI inference timed out after {inference_timeout}s")
        if state["error"]:
            ws.close()
            raise RuntimeError(state["error"])
        if not (state["finalized"] or {}).get("ok"):
            ws.close()
            raise RuntimeError((state["finalized"] or {}).get("message", "JoyAI recording failed"))
        try:
            with urllib.request.urlopen(f"{http_url}/download_last", timeout=120) as response:
                data = response.read()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
        finally:
            try:
                ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            ws.close()
        require_file(output_path, "JoyAI output video")
        return {"server": http_url, "managed_server": not bool(os.environ.get("JOYAI_SERVER_URL")), "frames_in": sent, "frames_out": state["frames_out"], "width": width, "height": height, "fps": fps, "num_inference_steps": num_inference_steps, "seed": seed, "reference_image": str(reference_image_path) if reference_image_path else None}

    def generate_video(self, video_path, prompt, output_path: Path, *, startup_timeout=1800, **kwargs) -> Dict[str, Any]:
        source = require_file(video_path, "source video")
        configured = os.environ.get("JOYAI_SERVER_URL")
        process = None
        log_handle = None
        with tempfile.TemporaryDirectory(prefix="joyai_v2v_") as tmp:
            work_dir = Path(tmp)
            if configured:
                ws_url, http_url = _service_urls(configured)
                self._wait_healthy(http_url, startup_timeout)
            else:
                ws_url, http_url, process, log_handle = self._start_server(work_dir, startup_timeout)
            try:
                return self._stream_video(ws_url, http_url, source, prompt, output_path, **kwargs)
            except Exception as exc:
                log_path = work_dir / "server.log"
                if process is not None and log_path.is_file():
                    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1600:]
                    raise RuntimeError(f"{exc}; JoyAI server log tail: {log_tail}") from exc
                raise
            finally:
                if process is not None:
                    self._stop_process(process)
                if log_handle is not None:
                    log_handle.close()


class JoyAIWrapper(ModelWrapper):
    def __init__(self, model="jd-opensource/JoyAI-Video-Edit", output_dir="./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = JoyAIService(model)

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
        return success_result(self.model, text_prompt, start, output, video_path, "joyai", metadata)
