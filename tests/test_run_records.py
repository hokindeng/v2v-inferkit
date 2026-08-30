"""run.py: every generation leaves a record; broken outputs never count as done."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as runmod  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "turntable_task" / "turntable_0000"


def _task(tmp_path, task_id="turntable_0000"):
    return {
        "id": task_id,
        "domain": "turntable",
        "domain_dir": "turntable_task",
        "prompt": "spin",
        "first_image_path": None,
        "final_image_path": None,
        "first_video_path": str(EXAMPLE / "first_video.mp4"),
        "ground_truth_video": str(EXAMPLE / "ground_truth.mp4"),
        "num_frames": 60,
    }


class FakeRunner:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0

    def run(self, model_name, image_path, text_prompt, question_data, **kwargs):
        self.calls += 1
        out = Path(kwargs.get("output_filename") or "")
        domain_dir = Path(self.output_dir) / question_data["domain_dir"]
        domain_dir.mkdir(parents=True, exist_ok=True)
        target = domain_dir / f"{question_data['id']}.mp4"
        if self.behaviour == "ok":
            shutil.copy(EXAMPLE / "first_video.mp4", target)
            return {"success": True, "video_path": str(target), "error": None, "duration_seconds": 1.0,
                    "generation_id": "req-ok", "model": model_name, "status": "success",
                    "metadata": {"payload": {"prompt": text_prompt}, "prompt_sent": text_prompt}}
        if self.behaviour == "empty":
            target.write_bytes(b"")
            return {"success": True, "video_path": str(target), "error": None, "duration_seconds": 1.0,
                    "generation_id": "req-empty", "model": model_name, "status": "success", "metadata": {}}
        if self.behaviour == "transient-then-ok":
            if self.calls == 1:
                return {"success": False, "video_path": None, "error": "503 Service Unavailable",
                        "duration_seconds": 1.0, "generation_id": "req-1", "model": model_name,
                        "status": "failed", "metadata": {}}
            shutil.copy(EXAMPLE / "first_video.mp4", target)
            return {"success": True, "video_path": str(target), "error": None, "duration_seconds": 1.0,
                    "generation_id": "req-2", "model": model_name, "status": "success", "metadata": {}}
        raise RuntimeError("boom")


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required")
def test_success_writes_record_with_ids_and_geometry(tmp_path):
    runner = FakeRunner("ok")
    runner.output_dir = tmp_path
    result = runmod.run_single_inference("m", _task(tmp_path), "turntable", tmp_path, runner=runner,
                                         controls={"seed": 1})
    assert result["success"] is True
    record = json.loads((tmp_path / "turntable_task" / "turntable_0000.json").read_text())
    assert record["generation_id"] == "req-ok"
    assert record["prompt_original"] == "spin"
    assert record["controls"] == {"seed": 1}
    assert record["output_geometry"]["frames"] > 0
    assert not (tmp_path / "turntable_task" / "turntable_0000.failed.json").exists()


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required")
def test_empty_output_is_a_failure_and_never_skipped(tmp_path):
    runner = FakeRunner("empty")
    runner.output_dir = tmp_path
    result = runmod.run_single_inference("m", _task(tmp_path), "turntable", tmp_path, runner=runner)
    assert result["success"] is False
    video, record, failed = runmod._record_paths(tmp_path, _task(tmp_path))
    assert failed.exists() and not record.exists() and not video.exists()
    assert runmod.output_is_complete(video, record) is False


def test_wrapper_exception_becomes_failed_record(tmp_path):
    runner = FakeRunner("raise")
    runner.output_dir = tmp_path
    result = runmod.run_single_inference("m", _task(tmp_path), "turntable", tmp_path, runner=runner)
    assert result["success"] is False
    failed = json.loads((tmp_path / "turntable_task" / "turntable_0000.failed.json").read_text())
    assert "boom" in failed["error"]


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required")
def test_experiment_retries_transient_errors_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(runmod.time, "sleep", lambda s: None)
    fake = FakeRunner("transient-then-ok")

    def fake_runner_factory(output_dir):
        fake.output_dir = Path(output_dir)
        return fake

    monkeypatch.setattr(runmod, "InferenceRunner", fake_runner_factory)
    monkeypatch.setitem(runmod.AVAILABLE_MODELS, "m", {"family": "F", "wrapper_module": "x"})
    tasks = {"turntable": [_task(tmp_path)]}
    out = runmod.run_experiment(tasks, {"m": "F"}, tmp_path, workers=1, retries=2)
    assert out["statistics"] == {"completed": 1, "failed": 0, "skipped": 0, "total_tasks": 1,
                                 "total_generations": 1, "duration_seconds": out["statistics"]["duration_seconds"]}
    assert fake.calls == 2
    manifests = list((tmp_path / "m").glob("run-*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["results"][0]["generation_id"] == "req-2"

    # second run: the valid mp4 + record are skipped
    out2 = runmod.run_experiment(tasks, {"m": "F"}, tmp_path, workers=1)
    assert out2["statistics"]["skipped"] == 1


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required")
def test_workers_do_not_collide(tmp_path, monkeypatch):
    fake = FakeRunner("ok")

    def fake_runner_factory(output_dir):
        fake.output_dir = Path(output_dir)
        return fake

    monkeypatch.setattr(runmod, "InferenceRunner", fake_runner_factory)
    monkeypatch.setitem(runmod.AVAILABLE_MODELS, "m", {"family": "F", "wrapper_module": "x"})
    tasks = {"turntable": [_task(tmp_path, f"turntable_{i:04d}") for i in range(8)]}
    out = runmod.run_experiment(tasks, {"m": "F"}, tmp_path, workers=4)
    assert out["statistics"]["completed"] == 8
    assert len(list((tmp_path / "m" / "turntable_task").glob("*.json"))) == 8
    assert len(list((tmp_path / "m" / "turntable_task").glob("*.mp4"))) == 8
