from pathlib import Path

import pytest

from v2vinferkit.models import fal_v2v_inference as module
from v2vinferkit.models.fal_v2v_inference import FalV2VService, FalV2VWrapper


@pytest.mark.parametrize(
    "profile,expected_field,expected_video,expected_reference,expected_duration,expected_resolution",
    [
        ("wan3", "reference_video_urls", ["https://input/video.mp4"], "Video 1", 6, "720p"),
        ("minimax_h3", "reference_video_urls", ["https://input/video.mp4"], "Video 1", 6, "768P"),
        ("seedance_2", "video_urls", ["https://input/video.mp4"], "@Video1", "6", "720p"),
        ("seedance_2_5", "video_urls", ["https://input/video.mp4"], "[Video1]", "6", "720p"),
    ],
)
def test_reference_payload_profiles(
    profile,
    expected_field,
    expected_video,
    expected_reference,
    expected_duration,
    expected_resolution,
):
    service = FalV2VService("provider/model", profile)
    payload = service.build_payload("turn the sky green", "https://input/video.mp4", 5.625)

    assert payload[expected_field] == expected_video
    assert expected_reference in payload["prompt"]
    assert payload["duration"] == expected_duration
    assert payload["resolution"] == expected_resolution


@pytest.mark.parametrize(
    "profile,expected_defaults",
    [
        ("gemini_omni_edit", {"video_url": "https://input/video.mp4", "resolution": "720p"}),
        ("kling_o3_edit", {"video_url": "https://input/video.mp4", "keep_audio": True}),
        (
            "happy_horse_edit",
            {"video_url": "https://input/video.mp4", "resolution": "720p", "audio_setting": "origin"},
        ),
        ("grok_edit", {"video_url": "https://input/video.mp4", "resolution": "720p"}),
    ],
)
def test_direct_edit_payload_profiles(profile, expected_defaults):
    service = FalV2VService("provider/model", profile)
    payload = service.build_payload("make it blue", "https://input/video.mp4", 5.0)

    for key, value in expected_defaults.items():
        assert payload[key] == value
    assert "duration" not in payload


def test_runtime_controls_override_profile_defaults():
    service = FalV2VService("provider/model", "seedance_2")
    payload = service.build_payload(
        "edit @Video1",
        "https://input/video.mp4",
        5.0,
        resolution="480p",
        aspect_ratio="16:9",
        generate_audio=False,
        seed=42,
    )

    assert payload["prompt"] == "edit @Video1"
    assert payload["resolution"] == "480p"
    assert payload["aspect_ratio"] == "16:9"
    assert payload["generate_audio"] is False
    assert payload["seed"] == 42


def test_profile_drops_controls_unsupported_by_endpoint():
    service = FalV2VService("provider/model", "gemini_omni_edit")
    payload = service.build_payload(
        "edit",
        "https://input/video.mp4",
        5.0,
        keep_audio=True,
        seed=42,
    )

    assert "keep_audio" not in payload
    assert "seed" not in payload


def test_explicit_duration_outside_profile_limit_fails():
    service = FalV2VService("provider/model", "seedance_2")
    with pytest.raises(ValueError, match="between 4 and 15"):
        service.build_payload(
            "edit",
            "https://input/video.mp4",
            5.0,
            requested_duration=16,
        )


def test_prepare_input_rejects_provider_truncation(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.touch()
    monkeypatch.setattr(module, "_probe_video_duration", lambda _: 8.5)

    service = FalV2VService("xai/grok-imagine-video/edit-video", "grok_edit")
    with pytest.raises(ValueError, match="at most 8s"):
        service.prepare_input(video)


def test_prepare_input_pads_short_video(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.touch()
    padded = tmp_path / "padded.mp4"
    padded.touch()
    monkeypatch.setattr(module, "_probe_video_duration", lambda _: 1.5)
    monkeypatch.setattr(module, "_pad_video", lambda *_: padded)

    service = FalV2VService("provider/model", "seedance_2")
    path, duration, is_temp = service.prepare_input(video)

    assert path == padded
    assert duration == 2.0
    assert is_temp is True


def test_queue_submit_polls_to_completion(monkeypatch):
    class InProgress:
        pass

    class Completed:
        pass

    class Handler:
        request_id = "req-123"

    class FakeFal:
        def __init__(self):
            self.statuses = iter([InProgress(), Completed()])

        def submit(self, endpoint, arguments):
            assert endpoint == "provider/model"
            assert arguments == {"prompt": "test"}
            return Handler()

        def status(self, endpoint, request_id, with_logs):
            assert request_id == "req-123"
            assert with_logs is True
            return next(self.statuses)

        def result(self, endpoint, request_id):
            return {"video": {"url": "https://output/video.mp4"}}

    service = FalV2VService("provider/model", "gemini_omni_edit", poll_interval=0)
    monkeypatch.setattr(service, "_fal_client", lambda: FakeFal())

    result, request_id = service.submit({"prompt": "test"})
    assert request_id == "req-123"
    assert result["video"]["url"] == "https://output/video.mp4"


def test_queue_timeout_retains_remote_request_id(monkeypatch):
    class InProgress:
        pass

    class Handler:
        request_id = "req-timeout"

    class FakeFal:
        def submit(self, endpoint, arguments):
            return Handler()

        def status(self, endpoint, request_id, with_logs):
            return InProgress()

    service = FalV2VService("provider/model", "gemini_omni_edit", poll_interval=0)
    monkeypatch.setattr(service, "_fal_client", lambda: FakeFal())

    with pytest.raises(module.FalRequestTimeout) as error:
        service.submit({"prompt": "test"}, max_wait=0)
    assert error.value.request_id == "req-timeout"


def test_wrapper_returns_standard_result(monkeypatch, tmp_path):
    wrapper = FalV2VWrapper(
        model="test-model",
        endpoint="provider/model",
        profile="gemini_omni_edit",
        output_dir=str(tmp_path),
    )

    def fake_generate_video(**kwargs):
        output_path = kwargs["output_path"]
        output_path.write_bytes(b"video")
        return {
            "video_path": str(output_path),
            "video_url": "https://output/video.mp4",
            "request_id": "req-123",
            "endpoint": "provider/model",
            "input_duration": 5.0,
            "payload": {"resolution": "720p"},
            "response": {"seed": 42},
        }

    monkeypatch.setattr(wrapper.service, "generate_video", fake_generate_video)
    result = wrapper.generate(None, "make it blue", video_path="input.mp4")

    assert set(result) == {
        "success",
        "video_path",
        "error",
        "duration_seconds",
        "generation_id",
        "model",
        "status",
        "metadata",
    }
    assert result["success"] is True
    assert result["generation_id"] == "req-123"
    assert Path(result["video_path"]).read_bytes() == b"video"
    assert result["metadata"]["seed"] == 42


def test_wrapper_failure_is_standardized(tmp_path):
    wrapper = FalV2VWrapper(
        model="test-model",
        endpoint="provider/model",
        profile="gemini_omni_edit",
        output_dir=str(tmp_path),
    )
    result = wrapper.generate(None, "make it blue")

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["video_path"] is None
    assert "video_path is required" in result["error"]
