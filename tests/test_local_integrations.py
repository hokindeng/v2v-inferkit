import importlib
import json
import queue
import subprocess
import sys
import types
from pathlib import Path

import pytest

from v2vinferkit.runner.MODEL_CATALOG import AVAILABLE_MODELS


LOCAL_MODEL_IDS = {
    model_id for model_id, config in AVAILABLE_MODELS.items()
    if config.get("deployment") == "local"
}


@pytest.mark.parametrize("model_id", sorted(LOCAL_MODEL_IDS))
def test_every_local_model_has_an_installer(model_id):
    root = Path(__file__).resolve().parents[1]
    script = root / "setup" / "models" / model_id / "setup.sh"
    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "model_id",
    [
        "wan-vace-1.3b-v2v",
        "joyai-video-edit-v2v",
        "bernini-r-1.3b-v2v",
        "bernini-r-14b-v2v",
        "kiwi-edit-5b-v2v",
        "editto-v2v",
        "sama-14b-v2v",
        "omnivideo2-1.3b-v2v",
        "omnivideo2-a14b-v2v",
        "coinve-edit-v2v",
        "lucy-edit-1.1-v2v",
    ],
)
def test_new_local_wrapper_imports_without_heavy_dependencies(model_id):
    config = AVAILABLE_MODELS[model_id]
    module = importlib.import_module(config["wrapper_module"])
    assert getattr(module, config["wrapper_class"])
    assert getattr(module, config["service_class"])


@pytest.mark.parametrize(
    ("model_id", "extra_args"),
    [
        ("joyai-video-edit-v2v", {}),
        ("bernini-r-1.3b-v2v", {}),
        ("kiwi-edit-5b-v2v", {}),
        ("editto-v2v", {}),
        ("sama-14b-v2v", {}),
        ("omnivideo2-1.3b-v2v", {"task": "v2v-1.3B"}),
        ("coinve-edit-v2v", {}),
        ("lucy-edit-1.1-v2v", {}),
    ],
)
def test_new_wrappers_reject_missing_video(tmp_path, model_id, extra_args):
    config = AVAILABLE_MODELS[model_id]
    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])
    wrapper = wrapper_class(model=config["model"], output_dir=str(tmp_path), **extra_args)
    result = wrapper.generate(None, "make the subject blue")
    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["video_path"] is None
    assert "video_path" in result["error"]


def test_bernini_builds_source_only_command(monkeypatch, tmp_path):
    from v2vinferkit.models import bernini_inference

    repo = tmp_path / "Bernini"
    repo.mkdir()
    (repo / "infer_single_gpu.py").touch()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = tmp_path / "input.mp4"
    source.touch()
    output = tmp_path / "output.mp4"
    captured = {}

    monkeypatch.setenv("BERNINI_REPO_PATH", str(repo))
    monkeypatch.setenv("BERNINI_WEIGHTS_PATH", str(checkpoint))

    def fake_run(command, **kwargs):
        captured["command"] = [str(part) for part in command]
        output.touch()

    monkeypatch.setattr(bernini_inference, "run_command", fake_run)
    service = bernini_inference.BerniniService("ByteDance/Bernini-R-1.3B-Diffusers")
    metadata = service.generate_video(source, "turn it blue", output, num_frames=82)

    assert "--guidance_mode" in captured["command"]
    assert captured["command"][captured["command"].index("--guidance_mode") + 1] == "v2v"
    assert "--images" not in captured["command"]
    assert metadata["num_frames"] == 81


def test_coinve_passes_multiple_instructions(monkeypatch, tmp_path):
    from v2vinferkit.models import coinve_inference

    repo = tmp_path / "CoinVE-200K"
    project = repo / "CoinVE-Edit"
    project.mkdir(parents=True)
    (project / "infer_coinve_single.py").touch()
    weights = tmp_path / "coinve"
    weights.mkdir()
    (weights / "coinve_edit_composite_vllm256_dit128.safetensors").touch()
    base = tmp_path / "base"
    (base / "Wan-AI" / "Wan2.1-T2V-14B").mkdir(parents=True)
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    source = tmp_path / "input.mp4"
    source.touch()
    output = tmp_path / "output.mp4"
    captured = {}

    monkeypatch.setenv("COINVE_REPO_PATH", str(repo))
    monkeypatch.setenv("COINVE_WEIGHTS_PATH", str(weights))
    monkeypatch.setenv("COINVE_BASE_MODELS_PATH", str(base))
    monkeypatch.setenv("COINVE_QWEN_PATH", str(qwen))

    def fake_run(command, **kwargs):
        captured["command"] = [str(part) for part in command]
        output.touch()

    monkeypatch.setattr(coinve_inference, "run_command", fake_run)
    service = coinve_inference.CoinVEService()
    metadata = service.generate_video(
        source,
        "unused fallback",
        output,
        instructions=["replace the car", "add a dog"],
    )
    assert metadata["instructions"] == ["replace the car", "add a dog"]
    prompt_index = captured["command"].index("--prompts")
    assert captured["command"][prompt_index + 1:prompt_index + 3] == ["replace the car", "add a dog"]


def test_joyai_url_normalization():
    from v2vinferkit.models.joyai_inference import _service_urls

    assert _service_urls("http://127.0.0.1:8080") == (
        "ws://127.0.0.1:8080/ws",
        "http://127.0.0.1:8080",
    )
    assert _service_urls("wss://example.test/ws") == (
        "wss://example.test/ws",
        "https://example.test",
    )


def test_joyai_streaming_protocol(monkeypatch, tmp_path):
    from v2vinferkit.models import joyai_inference

    class FakeWebSocket:
        def __init__(self):
            self.messages = queue.Queue()
            self.sent_json = []
            self.sent_frames = []

        def send(self, value):
            payload = json.loads(value)
            self.sent_json.append(payload)
            if payload["type"] == "start":
                self.messages.put(json.dumps({"type": "started"}))
            elif payload["type"] == "finalize_recording":
                self.messages.put(json.dumps({"type": "chunk_done", "frames_out": 2}))
                self.messages.put(json.dumps({"type": "recording_finalized", "ok": True}))

        def send_binary(self, value):
            self.sent_frames.append(value)

        def settimeout(self, value):
            self.timeout = value

        def recv(self):
            return self.messages.get(timeout=2)

        def close(self):
            self.messages.put(None)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"fake-mp4"

    fake_ws = FakeWebSocket()
    fake_module = types.SimpleNamespace(
        create_connection=lambda *args, **kwargs: fake_ws,
    )
    monkeypatch.setitem(sys.modules, "websocket", fake_module)
    monkeypatch.setattr(joyai_inference.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    service = joyai_inference.JoyAIService()
    monkeypatch.setattr(service, "_video_frames", lambda *args, **kwargs: iter([
        (1, 0.0, b"jpeg-one"),
        (2, 41.67, b"jpeg-two"),
    ]))
    source = tmp_path / "input.mp4"
    source.touch()
    output = tmp_path / "output.mp4"
    metadata = service._stream_video(
        "ws://localhost/ws",
        "http://localhost",
        source,
        "turn it blue",
        output,
        inference_timeout=2,
    )

    assert output.read_bytes() == b"fake-mp4"
    assert [message["type"] for message in fake_ws.sent_json] == [
        "start", "frame_meta", "frame_meta", "finalize_recording", "stop",
    ]
    assert fake_ws.sent_frames == [b"jpeg-one", b"jpeg-two"]
    assert metadata["frames_in"] == 2
    assert metadata["frames_out"] == 2
