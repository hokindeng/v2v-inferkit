import asyncio

import pytest

from v2vinferkit.models import runway_inference as rw


def test_overlong_prompt_is_refused_not_truncated(monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", "test")
    service = rw.RunwayService(model="aleph2")
    with pytest.raises(ValueError, match="Refusing rather than truncating"):
        asyncio.run(service.generate_video_to_video(prompt="x" * 1001, video_path="in.mp4"))


def test_runway_task_error_carries_task_id():
    err = rw.RunwayTaskError("task-1", "failed")
    assert err.task_id == "task-1"
    assert "task-1" in str(err)
