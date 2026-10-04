"""OmniVideo2 1.3B/A14B V2V editing, in-process on top of the official E2E inference script.

The upstream script (tools/inference/generate_omni_v2v[_1_3B].py) loads the DiT(s), T5, VAE and
Qwen3-VL-30B-A3B once and then loops over a prompt list. This wrapper imports that script as a module
and splits it in two: `_load` builds the pipeline exactly like the script's generate() does (args come
from the script's own argparse defaults), and `generate_video` runs the body of its per-prompt loop
(VAE encode -> Qwen3-VL caption + features -> T5 -> DiT sampling -> cache_video) with the upstream
helpers. Frames are sampled evenly across the whole clip with decord + the upstream transform (the
upstream reader only takes every k-th frame from the start). Components stay in CPU RAM and are moved to the GPU stage by stage, as upstream does, so the
A14B experts (2x28.6 GB) and the 62 GB VLM take turns on one 80 GB GPU. With V2V_IN_PROCESS=1 the
loaded pipeline is reused across tasks; without it each task's worker process loads it once.

Settings are the official launcher/README ones (tools/inference/inference_omni_v2v*.sh): 41 frames,
guide scale 3.0, shift 5, 40 unipc steps, 1.3B max_context_len 6272. The frame count matters: the DiT
context is [VLM | T5 | source latents] cut at max_context_len, and the source latents cost 390 tokens per
latent frame at 832x480, so 81 frames (21 latent frames, ~8.2k tokens) would truncate most of the source.
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, success_result, weights_path


MAX_FRAMES = int(os.environ.get("OMNIVIDEO2_FRAMES", "41"))  # official default; must be 4n+1


def _init_single_rank_group():
    """The upstream script always runs under torchrun with a process group; make a 1-rank one once."""
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    if dist.is_initialized():
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


class OmniVideo2Service:
    def __init__(self, model: str, task: str):
        self.model = model
        self.task = task
        self.repo = repo_path("Omni-Video", "OMNIVIDEO2_REPO_PATH")
        checkpoint_name = model.rsplit("/", 1)[-1]
        self.checkpoint = Path(os.environ.get("OMNIVIDEO2_WEIGHTS_PATH") or str(weights_path(checkpoint_name)))
        self.qwen = Path(os.environ.get("OMNIVIDEO2_QWEN_PATH") or str(weights_path("Qwen3-VL-30B-A3B-Instruct")))
        self.single = "1.3B" in task
        self.up = None          # the upstream script, imported as a module
        self.args = None        # its argparse namespace (upstream defaults)
        self.cfg = None
        self.pipe = None        # OmniVideoX2XUnified[1_3B]
        self.qwen_model = None  # loaded lazily on the first sample, as upstream does
        self.qwen_processor = None
        self.load_report = {}

    def _load(self):
        if self.pipe is not None:
            return
        repo = require_dir(self.repo, "OmniVideo2 repository")
        require_dir(self.checkpoint, "OmniVideo2 checkpoint")
        require_dir(self.qwen, "Qwen3-VL checkpoint")
        script = require_file(repo / "tools" / "inference" /
                              ("generate_omni_v2v_1_3B.py" if self.single else "generate_omni_v2v.py"),
                              "OmniVideo2 inference entrypoint")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # upstream launcher setting
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))  # the omnivideo package is not installed; upstream uses PYTHONPATH=<repo>
        spec = importlib.util.spec_from_file_location("omnivideo2_upstream_v2v", script)
        up = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(up)

        import torch

        argv = sys.argv
        sys.argv = [str(script), "--task", self.task, "--size", "832*480", "--ckpt_dir", str(self.checkpoint),
                    "--qwen3vl_model_path", str(self.qwen), "--base_seed", "42", "--frame_num", str(MAX_FRAMES),
                    "--sample_guide_scale", "3.0", "--sample_shift", "5"]  # official launcher values
        if self.single:
            sys.argv += ["--max_context_len", "6272"]  # inference_omni_v2v_1_3B.sh
        try:
            args = up._parse_args()  # remaining upstream defaults + _validate_args (checkpoint paths, 40 steps)
        finally:
            sys.argv = argv
        _init_single_rank_group()

        cfg = copy.deepcopy(up.WAN_CONFIGS["t2v-1.3B" if self.single else "t2v-A14B"])
        if self.single:
            checkpoints = {"model": args.new_checkpoint}
            pipe_cls = up.OmniVideoX2XUnified1_3B
        else:
            checkpoints = {"high_noise_model": args.new_checkpoint_high, "low_noise_model": args.new_checkpoint_low}
            pipe_cls = up.OmniVideoX2XUnified
        has_ckpt = any(p is not None and str(p).strip() for p in checkpoints.values())
        pipe = pipe_cls(
            config=cfg, checkpoint_dir=args.ckpt_dir, vlm_in_dim=cfg.vlm_in_dim, device_id=0, rank=0,
            use_usp=args.use_usp, t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp, sp_size=args.sp_size,
            use_visual_context_adapter=cfg.use_visual_context_adapter,
            visual_context_adapter_patch_size=cfg.visual_context_adapter_patch_size,
            max_context_len=args.max_context_len, init_on_cpu=True, wan_config=cfg if has_ckpt else None,
        )
        for attr, path in checkpoints.items():
            if path is None or not str(path).strip():
                continue
            state_dict = torch.load(path, map_location="cpu")
            state_dict = state_dict.get("module", state_dict.get("model", state_dict))
            for k in list(state_dict.keys()):
                if isinstance(state_dict[k], torch.Tensor):
                    state_dict[k] = state_dict[k].to(cfg.param_dtype)
            missing, unexpected = getattr(pipe, attr).load_state_dict(state_dict, strict=False)
            self.load_report[attr] = {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}
            del state_dict
            gc.collect()
        self.up, self.args, self.cfg, self.pipe = up, args, cfg, pipe

    def _read_source(self, source: str):
        """Evenly spaced frames over the whole clip -> (tensor [T,C,H,W], size, frame_num, out fps, info)."""
        import numpy as np

        up = self.up
        vr = up.decord.VideoReader(source)
        total, src_fps = len(vr), float(vr.get_avg_fps())
        h, w = vr[0].shape[:2]
        size = "480*832" if h > w else "832*480"  # both supported; the upstream reader skips mismatched orientations
        frame_num = max(1, ((min(self.args.frame_num, total) - 1) // 4) * 4 + 1)
        idx = np.linspace(0, total - 1, frame_num).round().astype(int).tolist()
        target_size = up.SIZE_CONFIGS[size]
        frames = up.transform_frames_to_tensor(vr.get_batch(idx).asnumpy(), (target_size[1], target_size[0]))
        out_fps = round(frame_num * src_fps / total, 3) if src_fps > 0 else 8  # output spans the source duration
        return frames, size, frame_num, out_fps, {"source_frames": total, "source_fps": round(src_fps, 3)}

    def _run_sample(self, source: str, prompt: str, output_path: Path) -> Dict[str, Any]:
        """Body of the upstream per-prompt loop for one (source, prompt) pair."""
        import torch

        up, args, cfg, pipe = self.up, self.args, self.cfg, self.pipe
        device = 0
        frames, size, frame_num, sample_fps, info = self._read_source(source)
        target_size = up.SIZE_CONFIGS[size]

        with torch.no_grad():
            frames = frames.to(device)
            pipe.vae.model.to(device)
            pipe.vae.mean = pipe.vae.mean.to(device)
            pipe.vae.std = pipe.vae.std.to(device)
            pipe.vae.scale = [pipe.vae.mean, 1.0 / pipe.vae.std]
            latent = pipe.vae.encode(frames.transpose(0, 1).unsqueeze(0))[0]
            pipe.vae.model.to("cpu")
            pipe.vae.mean = pipe.vae.mean.to("cpu")
            pipe.vae.std = pipe.vae.std.to("cpu")
            pipe.vae.scale = [pipe.vae.mean, 1.0 / pipe.vae.std]
            target_latent_frames = (frame_num - 1) // 4 + 1
            if latent.dim() == 4:
                latent = latent.unsqueeze(0)
            if latent.shape[2] > target_latent_frames:
                latent = latent[:, :, :target_latent_frames]
        visual_emb = latent
        del frames

        # Qwen3-VL: predicted target caption + last hidden states (DiT/T5/VAE parked on CPU meanwhile)
        up.offload_model_to_cpu(pipe)
        if self.qwen_model is None:
            self.qwen_model, self.qwen_processor = up.load_qwen3vl_model_and_processor(
                model_path=args.qwen3vl_model_path, device="cuda", dtype=args.qwen3vl_dtype,
                device_map=args.qwen3vl_device_map)
        else:
            up.load_qwen3vl_to_gpu(self.qwen_model, args.qwen3vl_device_map)
        caption, features = up.generate_caption_and_extract_features(
            model=self.qwen_model, processor=self.qwen_processor, source_video_path=source, edit_prompt=prompt,
            source_caption_system_prompt=cfg.source_caption_system_prompt,
            target_caption_system_prompt=cfg.target_caption_system_prompt,
            feature_extraction_system_prompt=cfg.feature_extraction_system_prompt,
            video_max_duration=args.video_max_duration, temperature=args.qwen3vl_temperature)
        ar_vision_input = features["vlm_last_hidden_states"].to(device)
        if ar_vision_input.dim() == 2:
            ar_vision_input = ar_vision_input.unsqueeze(0)
        del features
        up.offload_qwen3vl_to_cpu(self.qwen_model)
        up.load_model_to_gpu(pipe, device)

        # T5 on the predicted target caption and on the edit instruction
        t5_device = torch.device("cpu") if getattr(pipe, "t5_cpu", False) else device
        if not getattr(pipe, "t5_cpu", False):
            pipe.text_encoder.model.to(device)
        embs = []
        for text in (caption, prompt):
            emb = pipe.text_encoder([text], t5_device)
            embs.append((emb[0] if isinstance(emb, list) else emb).to(device))
        if not getattr(pipe, "t5_cpu", False):
            pipe.text_encoder.model.cpu()
        precomputed_context = torch.cat(embs, dim=0)

        vc_patch = cfg.visual_context_adapter_patch_size
        if isinstance(vc_patch, (list, tuple)) and vc_patch[0] > 1 and visual_emb.shape[2] % vc_patch[0]:
            pad = vc_patch[0] - visual_emb.shape[2] % vc_patch[0]
            visual_emb = torch.cat([visual_emb[:, :, 0:1].repeat(1, 1, pad, 1, 1), visual_emb], dim=2)

        video = pipe.generate(
            prompt, precomputed_context=precomputed_context, visual_emb=visual_emb, ar_vision_input=ar_vision_input,
            size=target_size, frame_num=frame_num, shift=args.sample_shift, sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps, guide_scale=args.sample_guide_scale, seed=args.base_seed,
            classifier_free_ratio=args.classifier_free_ratio, unconditioned_context=None,
            condition_mode=cfg.condition_mode, precision_dtype=cfg.param_dtype)
        if video is None:
            raise RuntimeError("OmniVideo2 returned no video")
        up.cache_video(tensor=video[None], save_file=str(output_path), fps=sample_fps, nrow=1, normalize=True,
                       value_range=(-1, 1))
        del video, visual_emb, ar_vision_input, precomputed_context, embs
        gc.collect()
        torch.cuda.empty_cache()
        return {"size": size, "num_frames": frame_num, "fps": sample_fps, **info, "target_caption": caption}

    def generate_video(self, video_path: Union[str, Path], prompt: str, output_path: Path, **_) -> Dict[str, Any]:
        source = require_file(video_path, "source video")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        try:
            sample = self._run_sample(str(source), prompt, output_path)
        except Exception:
            # park everything on CPU again so the next task starts from the same state
            self.up.offload_model_to_cpu(self.pipe)
            self.up.offload_qwen3vl_to_cpu(self.qwen_model)
            raise
        require_file(output_path, "OmniVideo2 output video")
        a = self.args
        return {"task": self.task, "checkpoint": str(self.checkpoint), "qwen_checkpoint": str(self.qwen),
                "frame_sampling": "even over the whole clip", "num_inference_steps": a.sample_steps,
                "guide_scale": a.sample_guide_scale, "shift": a.sample_shift, "solver": a.sample_solver,
                "seed": a.base_seed, "max_context_len": a.max_context_len,
                "vlm_video_max_duration": a.video_max_duration, "weights_load": self.load_report, **sample}


class OmniVideo2Wrapper(ModelWrapper):
    def __init__(self, model: str, output_dir: str = "./outputs", task: str = "v2v-A14B", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = OmniVideo2Service(model, task)

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
        return success_result(self.model, text_prompt, start, output, video_path, "omnivideo2", metadata)
