"""Lance 3B (ByteDance) instruction-guided video editing, official ``video_edit`` task, in-process.

The model is loaded once per Service (the same build as the upstream Gradio pipeline) and reused
across tasks. Input handling follows the model's training regime: frames are taken at ~12 fps
across the WHOLE source clip, up to the model's 121-frame maximum (longer clips are sampled
evenly, not truncated); the official loader then resizes to the 480p bucket nearest the source
aspect. The edit has the same frame count as the conditioning clip and is written at the fps that
makes its duration equal the source duration (audio dropped).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, success_result, weights_path

TRAIN_FPS = 12
MAX_FRAMES = 121  # model limit (10 s at 12 fps, 4k+1)
DEFAULT_STEPS = 30
DEFAULT_SHIFT = 3.5
DEFAULT_CFG_TEXT = 4.0


def _frame_count(duration_s: float) -> int:
    """Frames at ~12 fps over the clip, nearest 4k+1, capped at MAX_FRAMES, at least 5."""
    n = duration_s * TRAIN_FPS
    n = int(round((n - 1) / 4)) * 4 + 1
    return min(max(n, 5), MAX_FRAMES)


class LanceService:
    def __init__(self, model: str = "bytedance-research/Lance"):
        self.model = model
        self.repo = repo_path("Lance", "LANCE_REPO_PATH")
        self.weights = Path(os.environ.get("LANCE_WEIGHTS_PATH") or str(weights_path("Lance")))
        self._state: Optional[Dict[str, Any]] = None
        self._captured = []

    # ------------------------------------------------------------------ load
    def _load(self):
        if self._state is not None:
            return self._state
        repo = require_dir(self.repo, "Lance repository")
        ckpt = require_dir(self.weights / "Lance_3B_Video", "Lance_3B_Video checkpoint")
        vit_dir = require_dir(self.weights / "Qwen2.5-VL-ViT", "Qwen2.5-VL ViT checkpoint")
        vae_path = require_file(self.weights / "Wan2.2_VAE.pth", "Wan2.2 VAE checkpoint")
        os.environ.setdefault("POSITION_EMBEDDING_3D_VERSION", "v2")  # upstream sample_env.sh default
        os.environ.setdefault("EXP_HW_20250819", "False")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        import torch
        from safetensors.torch import load_file
        from transformers import set_seed
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

        import config.config_factory as cfg_factory
        if not str(Path(cfg_factory.__file__).resolve()).startswith(str(repo)):
            raise RuntimeError(f"'config' resolved outside the Lance repo: {cfg_factory.__file__}")
        # Point the upstream relative "downloads/..." paths at our weights (absolute, cwd-independent).
        paths = cfg_factory.get_model_path_config()
        paths["lance"]["video"] = str(ckpt)
        paths["vit"]["qwen2_5_vl"] = str(vit_dir)
        paths["vae"]["wan"] = str(vae_path)

        import inference_lance as il
        from config.config_factory import DataArguments, InferenceArguments, ModelArguments
        from data.data_utils import add_special_tokens
        from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.qwen2.modeling_qwen2 import Qwen2Config
        from modeling.vae.wan.model import WanVideoVAE
        from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

        device = 0
        torch.cuda.set_device(device)
        # Same arguments as inference_lance.sh / the upstream Gradio pipeline.
        model_args = ModelArguments(
            model_path=str(ckpt), vit_type="qwen_2_5_vl_original", llm_qk_norm=True, llm_qk_norm_und=True,
            llm_qk_norm_gen=True, tie_word_embeddings=False, max_num_frames=MAX_FRAMES, max_latent_size=64,
            latent_patch_size=[1, 1, 1],
        )
        model_args.vit_path = str(vit_dir)
        data_args = DataArguments()
        inf_args = InferenceArguments(
            validation_num_timesteps=DEFAULT_STEPS, validation_timestep_shift=DEFAULT_SHIFT, copy_init_moe=True,
            visual_und=True, visual_gen=True, vae_model_type="wan", apply_qwen_2_5_vl_pos_emb=True,
            apply_chat_template=False, cfg_type=0, validation_data_seed=42, num_frames=MAX_FRAMES,
            task="video_edit", save_path_gen=tempfile.gettempdir(), resolution="video_480p", text_template=True,
            use_KVcache=True,
        )
        il.apply_inference_defaults(model_args, data_args, inf_args)
        set_seed(inf_args.global_seed)

        llm_config = Qwen2Config.from_json_file(str(ckpt / "llm_config.json"))
        llm_config.layer_module = model_args.layer_module
        llm_config.qk_norm = model_args.llm_qk_norm
        llm_config.qk_norm_und = model_args.llm_qk_norm_und
        llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
        llm_config.tie_word_embeddings = model_args.tie_word_embeddings
        llm_config.freeze_und = inf_args.freeze_und
        llm_config.apply_qwen_2_5_vl_pos_emb = inf_args.apply_qwen_2_5_vl_pos_emb
        language_model = Qwen2ForCausalLM(llm_config)

        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(str(vit_dir))
        vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
        vit_weights = load_file(str(vit_dir / "vit.safetensors"))
        vit_model.load_state_dict(vit_weights, strict=True)
        il.clean_memory(vit_weights)

        vae_model = WanVideoVAE(device=torch.device("cuda", device))
        vae_config = deepcopy(vae_model.vae_config)

        config = LanceConfig(
            visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            latent_patch_size=model_args.latent_patch_size, max_num_frames=model_args.max_num_frames,
            max_latent_size=model_args.max_latent_size, vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
            connector_act=model_args.connector_act, interpolate_pos=model_args.interpolate_pos,
            timestep_shift=inf_args.timestep_shift,
        )
        model = Lance(language_model=language_model, vit_model=vit_model, vit_type=model_args.vit_type,
                      config=config, training_args=inf_args)
        model = model.to(dtype=torch.bfloat16)
        tokenizer = Qwen2Tokenizer.from_pretrained(str(ckpt))
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        language_model.init_moe()
        il.init_from_model_path_if_needed(model, model_args)
        if num_new_tokens > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)
        image_token_id = language_model.config.video_token_id
        new_token_ids.update({"image_token_id": image_token_id})
        model.update_tokenizer(tokenizer=tokenizer)
        model = model.to(device=device)
        model.eval()
        vae_model.eval() if hasattr(vae_model, "eval") else None

        # Capture the decoded THWC uint8 frames instead of letting upstream write a fixed-12-fps mp4.
        original_decode = il.decode_video_tensor

        def _capture(video_tensor, *args, **kwargs):
            kwargs["save_path"] = ""
            frames = original_decode(video_tensor, *args, **kwargs)
            self._captured.append(frames)
            return frames

        il.decode_video_tensor = _capture
        self._state = {
            "il": il, "torch": torch, "model": model, "vae": vae_model, "vae_config": vae_config,
            "tokenizer": tokenizer, "new_token_ids": new_token_ids, "image_token_id": image_token_id,
            "model_args": model_args, "data_args": data_args, "inf_args": inf_args, "ckpt": ckpt,
        }
        return self._state

    def _batch(self, st, prompt_file: Path, model_args, data_args, inf_args):
        from common.utils.misc import tuple_mul
        from data.dataset_base import DataConfig, simple_custom_collate
        from data.datasets_custom import ValidationDataset

        dc = DataConfig.from_yaml(str(prompt_file))
        dc.vit_patch_size = model_args.vit_patch_size
        dc.vit_patch_size_temporal = model_args.vit_patch_size_temporal
        dc.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        vc = st["vae_config"]
        dc.latent_patch_size = model_args.latent_patch_size
        dc.vae_downsample = tuple_mul(tuple(model_args.latent_patch_size),
                                      (vc.downsample_temporal, vc.downsample_spatial, vc.downsample_spatial))
        dc.max_latent_size = model_args.max_latent_size
        dc.max_num_frames = model_args.max_num_frames
        dc.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dc.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dc.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
        dc.num_frames = inf_args.num_frames
        dc.H = inf_args.video_height
        dc.W = inf_args.video_width
        dc.task = inf_args.task
        dc.resolution = inf_args.resolution
        dc.text_template = inf_args.text_template
        dc.enhance_prompt = False
        # Upstream clamps conditioning clips to 6 s; our prepared clip is <= 121 frames at 12 fps.
        dc.max_duration = MAX_FRAMES / TRAIN_FPS + 0.01
        ds = ValidationDataset(jsonl_path=str(prompt_file), tokenizer=st["tokenizer"], data_args=data_args,
                               model_args=model_args, training_args=inf_args, new_token_ids=st["new_token_ids"],
                               dataset_config=dc, local_rank=0, world_size=1)
        return simple_custom_collate([ds[0]])

    # -------------------------------------------------------------- generate
    def generate_video(self, video_path, prompt, output_path: Path, *, num_inference_steps: int = DEFAULT_STEPS,
                       guidance_scale: float = DEFAULT_CFG_TEXT, timestep_shift: float = DEFAULT_SHIFT,
                       seed: int = 42, max_frames: int = MAX_FRAMES) -> Dict[str, Any]:
        import decord
        import imageio
        import numpy as np

        source = require_file(video_path, "source video")
        reader = decord.VideoReader(str(source), ctx=decord.cpu(0))
        src_count = len(reader)
        src_fps = float(reader.get_avg_fps()) or 24.0
        if src_count < 1:
            raise ValueError(f"No frames decoded from {source}")
        src_duration = src_count / src_fps
        count = min(_frame_count(src_duration), ((int(max_frames) - 1) // 4) * 4 + 1)
        idx = [round(i * (src_count - 1) / max(count - 1, 1)) for i in range(count)]
        frames = reader.get_batch(idx).asnumpy()
        del reader
        src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
        out_fps = count / src_duration

        st = self._load()
        il, torch = st["il"], st["torch"]
        with tempfile.TemporaryDirectory(prefix="lance_edit_") as tmp:
            tmp = Path(tmp)
            # Conditioning clip: the evenly sampled frames at the training fps, so the upstream loader
            # (12 fps sampler) keeps every one of them.
            clip = tmp / "source.mp4"
            imageio.mimsave(str(clip), list(frames), fps=TRAIN_FPS, format="mp4", quality=9, macro_block_size=2)
            prompt_file = tmp / "request.json"
            prompt_file.write_text(json.dumps({"000000": {
                "interleave_array": [prompt, str(clip), str(clip)],
                "element_dtype_array": ["text", "video", "video"],
                "istarget_in_interleave": [0, 0, 1],
            }}, ensure_ascii=False), encoding="utf-8")

            model_args = deepcopy(st["model_args"])
            model_args.cfg_text_scale = float(guidance_scale)
            data_args = deepcopy(st["data_args"])
            data_args.val_dataset_config_file = str(prompt_file)
            inf_args = deepcopy(st["inf_args"])
            inf_args.validation_num_timesteps = int(num_inference_steps)
            inf_args.validation_timestep_shift = float(timestep_shift)
            inf_args.validation_data_seed = int(seed)
            inf_args.validation_noise_seed = int(seed)
            inf_args.num_frames = count
            inf_args.save_path_gen = str(tmp)
            inf_args.prompt_data_dict = {}

            batch = self._batch(st, prompt_file, model_args, data_args, inf_args)
            self._captured.clear()
            il.clean_memory()
            il.validate_on_fixed_batch(
                fsdp_model=st["model"], vae_model=st["vae"], tokenizer=st["tokenizer"], val_data_cpu=batch,
                training_args=inf_args, model_args=model_args, inference_args=inf_args,
                new_token_ids=st["new_token_ids"], image_token_id=st["image_token_id"], device=0,
                save_source_video=False, save_path_gen=str(tmp), save_path_gt="",
            )
            del batch
        if not self._captured:
            raise RuntimeError("Lance returned no decoded video")
        out = np.asarray(self._captured[-1])
        self._captured.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if out.ndim != 4 or out.shape[0] < 2:
            raise RuntimeError(f"Unexpected Lance output shape {getattr(out, 'shape', None)}")
        out_fps = out.shape[0] / src_duration
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), list(out), fps=out_fps, format="mp4", quality=9, macro_block_size=1)
        return {
            "task": "video_edit", "checkpoint": f"{self.model}/Lance_3B_Video", "num_inference_steps": int(num_inference_steps),
            "cfg_text_scale": float(guidance_scale), "timestep_shift": float(timestep_shift), "kv_cache": True,
            "seed": int(seed), "resolution_preset": "video_480p", "source_frames": src_count, "source_fps": round(src_fps, 3),
            "source_size": [src_w, src_h], "source_duration_s": round(src_duration, 3), "num_frames": int(out.shape[0]),
            "sampled_fps": round(count / src_duration, 3), "subsampled": count < round(src_duration * TRAIN_FPS),
            "height": int(out.shape[1]), "width": int(out.shape[2]),
            "fps": round(out_fps, 4),
        }


class LanceWrapper(ModelWrapper):
    def __init__(self, model="bytedance-research/Lance", output_dir="./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = LanceService(model)

    def generate(self, image_path, text_prompt, duration=5.0, output_filename=None, video_path=None, **kwargs):
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "video_path is required")
        kwargs.pop("question_data", None)
        gt_frames = kwargs.pop("num_frames", None)  # editing covers the input clip; GT length is not a target
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            metadata = self.service.generate_video(video_path, text_prompt, output, **kwargs)
        except Exception as exc:
            return failed_result(self.model, text_prompt, start, exc, video_path)
        metadata["ground_truth_frames"] = gt_frames
        return success_result(self.model, text_prompt, start, output, video_path, "lance", metadata)
