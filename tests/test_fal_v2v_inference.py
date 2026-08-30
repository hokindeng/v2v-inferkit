from pathlib import Path

import pytest

from v2vinferkit.models import fal_v2v_inference as module
from v2vinferkit.models.fal_v2v_inference import FalV2VService, FalV2VWrapper


@pytest.mark.parametrize(
    "profile,expected_field,expected_video,expected_duration,expected_resolution,audio_key",
    [
        ("wan3", "reference_video_urls", ["https://input/video.mp4"], 6, "720p", "audio"),
        ("minimax_h3", "reference_video_urls", ["https://input/video.mp4"], 6, "768P", None),
        ("seedance_2", "video_urls", ["https://input/video.mp4"], "6", "720p", "generate_audio"),
        ("seedance_2_5", "video_urls", ["https://input/video.mp4"], "6", "720p", "generate_audio"),
    ],
)
def test_reference_payload_profiles(
    profile,
    expected_field,
    expected_video,
    expected_duration,
    expected_resolution,
    audio_key,
):
    service = FalV2VService("provider/model", profile)
    payload = service.build_payload("turn the sky green", "https://input/video.mp4", 5.625)

    assert payload[expected_field] == expected_video
    # The benchmark prompt goes out verbatim: no reference labels, no "apply this edit".
    assert payload["prompt"] == "turn the sky green"
    assert payload["duration"] == expected_duration
    assert payload["resolution"] == expected_resolution
    if audio_key:
        assert payload[audio_key] is False
    assert payload.get("enable_prompt_expansion") in (None, False)


BENCHMARK_SOURCE_SECONDS = 2.5  # 60 frames @ 24 fps


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("seedance_2_5", {"duration": "4", "generate_audio": False, "resolution": "720p"}),
        ("wan3", {"duration": 3, "audio": False, "enable_prompt_expansion": False}),
        ("minimax_h3", {"duration": 5, "resolution": "768P"}),
        ("kling_o3_edit", {"keep_audio": False}),
        ("veo31_extend", {"duration": "7s", "resolution": "720p", "generate_audio": False, "aspect_ratio": "16:9"}),
    ],
)
def test_benchmark_payload_snapshots(profile, expected):
    """What each in-scope endpoint receives for a 2.5 s silent source, prompt verbatim."""
    service = FalV2VService("provider/model", profile)
    payload = service.build_payload("P", "https://input/video.mp4", BENCHMARK_SOURCE_SECONDS, seed=7)

    assert payload["prompt"] == "P"
    for key, value in expected.items():
        assert payload[key] == value
    if "seed" in service.profile.get("allowed_controls", set()):
        assert payload["seed"] == 7


def test_ltx_extend_sends_exact_float_duration_and_context():
    service = FalV2VService("fal-ai/ltx-2.3/extend-video", "ltx23_extend")
    payload = service.build_payload("P", "https://input/video.mp4", 2.5, requested_duration=2.5)

    assert payload["duration"] == 2.5
    assert payload["mode"] == "end"
    assert payload["context"] == 2.5
    with pytest.raises(ValueError, match="between 2 and 20"):
        service.build_payload("P", "https://input/video.mp4", 2.5, requested_duration=1.0)


def test_veo_extend_refuses_requested_duration():
    service = FalV2VService("fal-ai/veo3.1/extend-video", "veo31_extend")
    with pytest.raises(ValueError, match="fixed output duration"):
        service.build_payload("P", "https://input/video.mp4", 2.5, requested_duration=3)


def test_grok_extend_rejects_out_of_range_extension():
    service = FalV2VService("xai/grok-imagine-video/extend-video", "grok_extend")
    with pytest.raises(ValueError, match="one of 2, 3"):
        service.build_payload("P", "https://input/video.mp4", 2.5, requested_duration=12)


@pytest.mark.parametrize(
    "profile,expected_defaults",
    [
        (
            "luma_ray_3_2",
            {"video_url": "https://input/video.mp4", "resolution": "720p", "duration": "5s"},
        ),
        ("kling_o1_edit", {"video_url": "https://input/video.mp4", "keep_audio": True}),
        (
            "wan27_edit",
            {"video_url": "https://input/video.mp4", "resolution": "1080p", "duration": 0},
        ),
        ("gemini_omni", {"video_url": "https://input/video.mp4"}),
        ("gemini_omni_edit", {"video_url": "https://input/video.mp4", "resolution": "720p"}),
        ("kling_o3_edit", {"video_url": "https://input/video.mp4", "keep_audio": False}),
        (
            "happy_horse_edit",
            {"video_url": "https://input/video.mp4", "resolution": "720p", "audio_setting": "origin"},
        ),
        ("grok_edit", {"video_url": "https://input/video.mp4", "resolution": "720p"}),
        ("grok_extend", {"video_url": "https://input/video.mp4", "duration": 6}),
    ],
)
def test_direct_edit_payload_profiles(profile, expected_defaults):
    service = FalV2VService("provider/model", profile)
    payload = service.build_payload("make it blue", "https://input/video.mp4", 5.0)

    for key, value in expected_defaults.items():
        assert payload[key] == value
    if "duration" not in expected_defaults:
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


@pytest.mark.parametrize(
    "profile,duration,expected",
    [
        ("luma_ray_3_2", 10, "10s"),
        ("wan27_edit", 6, 6),
        ("grok_extend", 2.5, 3),
    ],
)
def test_discrete_duration_profiles(profile, duration, expected):
    service = FalV2VService("provider/model", profile)
    payload = service.build_payload(
        "edit",
        "https://input/video.mp4",
        5.0,
        requested_duration=duration,
    )

    assert payload["duration"] == expected


def test_discrete_duration_profile_rejects_unsupported_value():
    service = FalV2VService("provider/model", "luma_ray_3_2")
    with pytest.raises(ValueError, match="one of 5, 10 seconds"):
        service.build_payload(
            "edit",
            "https://input/video.mp4",
            5.0,
            requested_duration=6,
        )


def test_extension_length_covers_ground_truth(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.touch()
    monkeypatch.setattr(module, "_probe_video_fps", lambda _: 24.0)

    # 60 reference frames at 24 fps = 2.5 s -> rounded up to 3 s
    assert module.extension_seconds(video, 60) == 3
    assert module.extension_seconds(video, 24) == 1


def test_extend_wrapper_derives_duration_from_num_frames(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.touch()
    monkeypatch.setattr(module, "_probe_video_fps", lambda _: 24.0)
    captured = {}

    def fake_generate_video(prompt, video_path, output_path, duration=None, max_wait=None, **kwargs):
        captured["duration"] = duration
        return {
            "video_path": str(output_path),
            "video_url": "https://out/video.mp4",
            "request_id": "req-1",
            "endpoint": "xai/grok-imagine-video/extend-video",
            "payload": {"duration": duration},
            "input_duration": 2.5,
            "response": {},
        }

    wrapper = FalV2VWrapper(
        model="grok-imagine-video-extend",
        endpoint="xai/grok-imagine-video/extend-video",
        profile="grok_extend",
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(wrapper.service, "generate_video", fake_generate_video)

    result = wrapper.generate(text_prompt="continue", video_path=video, num_frames=60)

    assert result["success"] is True
    assert captured["duration"] == 3


def test_extend_wrapper_refuses_to_guess_extension_length(monkeypatch, tmp_path):
    """No ground truth -> no request. Falling back to the provider default (6 s) would double the bill."""
    video = tmp_path / "input.mp4"
    video.touch()
    wrapper = FalV2VWrapper(
        model="grok-imagine-video-extend",
        endpoint="xai/grok-imagine-video/extend-video",
        profile="grok_extend",
        output_dir=str(tmp_path),
    )
    called = {"n": 0}
    monkeypatch.setattr(wrapper.service, "generate_video", lambda **kw: called.__setitem__("n", 1))

    result = wrapper.generate(text_prompt="continue", video_path=video)

    assert result["success"] is False
    assert "extension length unknown" in result["error"]
    assert called["n"] == 0


def test_ltx_wrapper_derives_float_extension(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.touch()
    monkeypatch.setattr(module, "_probe_video_fps", lambda _: 24.0)
    captured = {}

    def fake_generate_video(prompt, video_path, output_path, duration=None, max_wait=None, **kwargs):
        captured["duration"] = duration
        return {"video_path": str(output_path), "video_url": "u", "request_id": "r",
                "endpoint": "e", "payload": {"duration": duration}, "input_duration": 2.5, "response": {}}

    wrapper = FalV2VWrapper(model="ltx-2.3-extend", endpoint="fal-ai/ltx-2.3/extend-video",
                            profile="ltx23_extend", output_dir=str(tmp_path))
    monkeypatch.setattr(wrapper.service, "generate_video", fake_generate_video)
    result = wrapper.generate(text_prompt="continue", video_path=video, num_frames=60)

    assert result["success"] is True
    assert captured["duration"] == 2.5


def test_extend_input_guard_matches_endpoint_limits(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    video.touch()
    monkeypatch.setattr(module, "_probe_video_duration", lambda _: 15.5)

    service = FalV2VService("xai/grok-imagine-video/extend-video", "grok_extend")
    with pytest.raises(ValueError, match="at most 15s"):
        service.prepare_input(video)


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


def test_pad_video_clones_first_frame_in_front(monkeypatch, tmp_path):
    captured = {}

    class Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return Done()

    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(module, "_has_audio_stream", lambda _: False)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._pad_video(tmp_path / "input.mp4", 2.5, 3.0)

    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert vf == "tpad=start_mode=clone:start_duration=0.500"
    assert "stop_mode" not in vf


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


def test_queue_timeout_cancels_and_retains_remote_request_id(monkeypatch):
    class InProgress:
        pass

    class Handler:
        request_id = "req-timeout"

    cancelled = []

    class FakeFal:
        def submit(self, endpoint, arguments):
            return Handler()

        def status(self, endpoint, request_id, with_logs):
            return InProgress()

        def cancel(self, endpoint, request_id):
            cancelled.append(request_id)

    service = FalV2VService("provider/model", "gemini_omni_edit", poll_interval=0)
    monkeypatch.setattr(service, "_fal_client", lambda: FakeFal())

    with pytest.raises(module.FalRequestTimeout) as error:
        service.submit({"prompt": "test"}, max_wait=0)
    assert error.value.request_id == "req-timeout"
    assert error.value.cancelled is True
    assert cancelled == ["req-timeout"]


def test_completed_with_error_is_a_traceable_failure(monkeypatch):
    """fal_client has no Failed status: a failed job is Completed(error=...)."""

    class Completed:
        error = "content policy violation"
        error_type = "moderation"

    class Handler:
        request_id = "req-moderated"

    class FakeFal:
        def submit(self, endpoint, arguments):
            return Handler()

        def status(self, endpoint, request_id, with_logs):
            return Completed()

        def result(self, endpoint, request_id):
            raise AssertionError("result() must not be called on an errored job")

    service = FalV2VService("provider/model", "gemini_omni_edit", poll_interval=0)
    monkeypatch.setattr(service, "_fal_client", lambda: FakeFal())

    with pytest.raises(module.FalRequestFailed) as error:
        service.submit({"prompt": "test"})
    assert error.value.request_id == "req-moderated"
    assert "moderation" in str(error.value)


def test_result_http_error_keeps_request_id(monkeypatch):
    class Completed:
        pass

    class Handler:
        request_id = "req-http"

    class FakeFal:
        def submit(self, endpoint, arguments):
            return Handler()

        def status(self, endpoint, request_id, with_logs):
            return Completed()

        def result(self, endpoint, request_id):
            raise RuntimeError("422 Unprocessable")

    service = FalV2VService("provider/model", "gemini_omni_edit", poll_interval=0)
    monkeypatch.setattr(service, "_fal_client", lambda: FakeFal())

    with pytest.raises(module.FalRequestFailed) as error:
        service.submit({"prompt": "test"})
    assert error.value.request_id == "req-http"
    assert "422" in str(error.value)


def test_download_never_leaves_an_empty_file(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            pass

        def iter_bytes(self):
            return iter([])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Client:
        def __init__(self, **kw):
            pass

        def stream(self, method, url):
            return Response()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(module.httpx, "Client", Client)
    target = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="0 bytes"):
        FalV2VService.download("https://x/video.mp4", target)
    assert not target.exists()
    assert not target.with_name("out.mp4.part").exists()


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
