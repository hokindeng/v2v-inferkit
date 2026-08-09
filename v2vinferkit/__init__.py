"""v2v-inferkit — unified inference toolkit for video-to-video models.

Ten V2V models — five commercial APIs and five open-source — that take a
conditioning video plus a text prompt and return an edited video.
"""

__version__ = "0.1.0"

__all__ = ["run_inference", "InferenceRunner", "__version__"]


def __getattr__(name: str):
    """Lazy import so listing models never loads provider SDKs."""
    if name in ("run_inference", "InferenceRunner"):
        from .runner.inference import run_inference, InferenceRunner
        return {"run_inference": run_inference, "InferenceRunner": InferenceRunner}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
