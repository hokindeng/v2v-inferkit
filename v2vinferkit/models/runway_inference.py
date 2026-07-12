"""
Runway ML Image-to-Video Generation Service.

Supports text + image → video generation using Runway's Gen-3, Gen-4, and Gen-4.5 models.

Image Upload: Uses Runway's ephemeral upload feature (no S3/external hosting required).
Images are uploaded directly to Runway's storage and get a runway:// URI valid for 24 hours.
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
import io
from PIL import Image
from dotenv import load_dotenv
from .base import ModelWrapper
from ..utils.image import load_image_rgb

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RunwayService:
    """Service for image-to-video generation using Runway ML models."""
    
    def __init__(self, model: str = "gen4_turbo"):
        """
        Initialize Runway service.
        
        Args:
            model: Runway model to use (gen4.5, gen4_turbo; aleph2 for v2v)
        """
        self.api_secret = os.getenv("RUNWAYML_API_SECRET")
        if not self.api_secret:
            raise ValueError("RUNWAYML_API_SECRET environment variable is required")
        
        self.model = model
        
        # Validate model and set constraints
        self.model_constraints = self._get_model_constraints(model)
    
    def _get_model_constraints(self, model: str) -> Dict[str, Any]:
        """Get model-specific constraints."""
        constraints = {
            "gen4.5": {
                "durations": [5, 10],
                # Gen-4.5 supports same ratios as Gen-4 with enhanced quality
                "ratios": ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"],
                "description": "Runway Gen-4.5 - World's top-rated video model with unprecedented visual fidelity"
            },
            "gen4_turbo": {
                "durations": [5, 10],
                # Use actual pixel dimensions required by Runway API
                "ratios": ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"],
                "description": "Runway Gen-4 Turbo - Fast high-quality generation"
            },
            # `gen4_aleph` (i2v) removed 2026-06-22 — deprecated by the Runway API
            # (400 invalid model id). Aleph is video-to-video only; see `aleph2` below.
            "aleph2": {
                # Aleph is a video-to-video model: it edits/continues an input video.
                # Output length follows the input video; the API rejects a `duration` key,
                # so durations is intentionally empty. Ratios per runwayml SDK 5.2.0
                # video_to_video.create() for model='aleph2'.
                "durations": [],
                "ratios": ["1280:720", "720:1280", "1112:834", "834:1112", "960:960",
                           "1470:630", "992:432", "864:496", "752:560", "640:640",
                           "560:752", "496:864"],
                "description": "Runway Aleph 2 - Video-to-video (text + video -> video)"
            },
        }
        
        if model not in constraints:
            raise ValueError(f"Unknown Runway model: {model}. Available: {list(constraints.keys())}")
        
        return constraints[model]
    
    def _determine_best_aspect_ratio(self, image_width: int, image_height: int) -> str:
        """
        Determine the best aspect ratio match from supported ratios.
        
        Args:
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Best matching aspect ratio string (e.g., "960:960", "1280:720")
        """
        input_ratio = image_width / image_height
        supported_ratios = self.model_constraints["ratios"]
        
        # Check for square images and use 960:960 if available
        if 0.9 <= input_ratio <= 1.1:
            # Look for square format in supported ratios
            for ratio_str in supported_ratios:
                if ':' in ratio_str:
                    parts = ratio_str.split(':')
                    w, h = map(int, parts)
                    if w == h:  # Square format found
                        logger.info(f"Square image detected ({image_width}×{image_height}) -> using {ratio_str}")
                        return ratio_str
        
        best_ratio = None
        min_diff = float('inf')
        
        for ratio_str in supported_ratios:
            # Skip 1:1 if we already handled it above
            if ratio_str == "1:1":
                continue
                
            # Handle both aspect ratio strings (e.g., "16:9") and pixel dimensions (e.g., "1280:768")
            if ':' in ratio_str:
                parts = ratio_str.split(':')
                w, h = map(float, parts)  # Use float to handle both "16:9" and "1280:768"
                ratio = w / h
            else:
                # Fallback for any other format
                continue
            
            diff = abs(input_ratio - ratio)
            
            if diff < min_diff:
                min_diff = diff
                best_ratio = ratio_str
        
        logger.info(f"Input aspect ratio {input_ratio:.3f} ({image_width}×{image_height}) -> Best match: {best_ratio}")
        return best_ratio

    def _ensure_min_duration(self, video_path: str, min_seconds: float = 2.0) -> str:
        import subprocess, tempfile, math
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=nb_frames,r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in probe.stdout.strip().split("\n") if l.strip()]
        duration = float(lines[-1]) if lines else 0
        if duration >= min_seconds:
            return video_path
        fps_str, nb_frames = None, None
        for line in lines:
            if "/" in line:
                parts = line.split(",")
                fps_str = parts[0]
                if len(parts) > 1:
                    nb_frames = int(parts[1])
            elif line.replace(".", "").isdigit() and "." in line:
                pass
        if not nb_frames or nb_frames <= 0:
            return video_path
        target_fps = max(1, math.floor(nb_frames / min_seconds))
        num, den = (int(x) for x in fps_str.split("/")) if fps_str and "/" in fps_str else (12, 1)
        orig_fps = num / den
        slow_factor = orig_fps / target_fps
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-filter:v", f"setpts=PTS*{slow_factor}",
             "-r", str(target_fps),
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", out],
            capture_output=True
        )
        logger.info(f"Stretched {nb_frames} frames from {duration:.1f}s to {nb_frames/target_fps:.1f}s (fps {orig_fps}->{target_fps})")
        return out

    def _resize_and_pad_image(self, image_path: Union[str, Path], target_ratio: str) -> Path:
        """
        Resize and pad image to match target dimensions exactly.
        
        Args:
            image_path: Path to input image
            target_ratio: Target aspect ratio (e.g., "16:9" or "1280:768")
            
        Returns:
            Path to processed image file
        """
        # Parse target dimensions - handle both aspect ratios and pixel dimensions
        if ':' in target_ratio:
            parts = target_ratio.split(':')
            if '.' not in parts[0] and len(parts[0]) > 2:  # Likely pixel dimensions like "1280:768"
                target_w, target_h = map(int, parts)
            else:  # Aspect ratio like "16:9"
                # Convert aspect ratio to target dimensions (use standard 720p resolution)
                aspect_w, aspect_h = map(float, parts)
                aspect_ratio = aspect_w / aspect_h
                if aspect_ratio > 1:  # Landscape
                    target_w, target_h = 1280, int(1280 / aspect_ratio)
                else:  # Portrait
                    target_h, target_w = 1280, int(1280 * aspect_ratio)
        else:
            # Default to 720p 16:9 if format not recognized
            target_w, target_h = 1280, 720
        
        # Load and convert image
        image = load_image_rgb(image_path)

        original_w, original_h = image.size
        logger.info(f"Original image size: {original_w}×{original_h}")
        
        # Calculate scaling to fit image within target dimensions while preserving aspect ratio
        scale_w = target_w / original_w
        scale_h = target_h / original_h
        scale = min(scale_w, scale_h)  # Use smaller scale to ensure image fits
        
        # Resize image
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create canvas with exact target dimensions and white background
        padded_image = Image.new("RGB", (target_w, target_h), color="white")
        
        # Calculate position to center the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Paste resized image onto padded canvas
        padded_image.paste(resized_image, (x_offset, y_offset))
        
        # Save processed image
        processed_path = Path(image_path).parent / f"runway_processed_{Path(image_path).name}"
        padded_image.save(processed_path, "PNG", quality=95)
        
        logger.info(f"Processed image: {original_w}×{original_h} -> {new_w}×{new_h} -> {target_w}×{target_h}")
        logger.info(f"Saved to: {processed_path}")
        
        return processed_path
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        duration: int = 5,
        ratio: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt and image.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file (will be uploaded to get URL)
            duration: Video duration in seconds
            ratio: Video aspect ratio (if not provided, uses first available)
            output_path: Optional path to save video
            
        Returns:
            Dictionary with generation results
        """
        # Validate duration
        if duration not in self.model_constraints["durations"]:
            valid_durations = self.model_constraints["durations"]
            logger.warning(f"Duration {duration}s not supported for {self.model}. Using {valid_durations[0]}s")
            duration = valid_durations[0]
        
        # Determine best aspect ratio if not provided
        if not ratio:
            # Load image to get dimensions for aspect ratio detection
            with Image.open(image_path) as img:
                ratio = self._determine_best_aspect_ratio(img.width, img.height)
        elif ratio not in self.model_constraints["ratios"]:
            valid_ratios = self.model_constraints["ratios"]
            logger.warning(f"Ratio {ratio} not supported for {self.model}. Using {valid_ratios[0]}")
            ratio = valid_ratios[0]
        
        # Runway limits prompt_text to 1000 characters
        if len(prompt) > 1000:
            prompt = prompt[:997] + "..."

        # Process image to match target dimensions
        processed_image_path = self._resize_and_pad_image(image_path, ratio)
        
        # Upload processed image to get URL (Runway requires image URLs)
        image_url = await self._upload_image(processed_image_path)
        
        try:
            # Generate video using Runway SDK
            result = await self._generate_with_runway(prompt, image_url, duration, ratio)
            
            # Download video if output path provided
            if output_path and result.get("video_url"):
                saved_path = await self._download_video(result["video_url"], output_path)
                result["video_path"] = str(saved_path)
                logger.info(f"Video saved to: {saved_path}")
            
            result.update({
                "model": self.model,
                "prompt": prompt,
                "image_path": str(image_path),
                "processed_image_path": str(processed_image_path),
                "duration": duration,
                "ratio": ratio
            })
            
            return result
            
        finally:
            # Clean up temporary processed image
            try:
                if processed_image_path.exists():
                    processed_image_path.unlink()
                    logger.debug(f"Cleaned up processed image: {processed_image_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up processed image: {e}")
    
    async def generate_video_to_video(
        self,
        prompt: str,
        video_path: Union[str, Path],
        duration: Optional[int] = None,
        ratio: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate a video from text prompt + input video (Aleph video-to-video).

        Aleph edits/continues the input video under text guidance. The input video
        defines the spatial dimensions, so `ratio` is optional — only forwarded when
        explicitly provided and supported.

        Args:
            prompt: Text instructions for the video edit/continuation.
            video_path: Path to the input video (uploaded via Runway ephemeral upload).
            duration: Optional output duration in seconds.
            ratio: Optional output ratio (model-specific pixel dimensions).
            output_path: Optional local path to save the result.

        Returns:
            Dict with task_id, video_url, status (+ video_path when downloaded).
        """
        if ratio and ratio not in self.model_constraints["ratios"]:
            logger.warning(f"Ratio {ratio} not supported for {self.model}; letting Aleph infer from the input video")
            ratio = None
        # NOTE: Aleph (aleph2) rejects a `duration` key — the output length follows the
        # input video. `duration` is accepted/ignored here only for interface symmetry
        # with i2v; it is never forwarded to the v2v endpoint.

        if len(prompt) > 1000:
            prompt = prompt[:997] + "..."

        adapted_path = self._ensure_min_duration(str(video_path), min_seconds=2.0)
        try:
            video_uri = await self._upload_file(adapted_path)

            result = await self._generate_v2v_with_runway(prompt, video_uri, ratio)

            if output_path and result.get("video_url"):
                saved_path = await self._download_video(result["video_url"], output_path)
                result["video_path"] = str(saved_path)
                logger.info(f"Video saved to: {saved_path}")

            result.update({
                "model": self.model,
                "prompt": prompt,
                "video_path_input": str(video_path),
                "ratio": ratio,
            })
            return result
        finally:
            if adapted_path != str(video_path):
                try:
                    os.unlink(adapted_path)
                except OSError:
                    pass

    async def _generate_v2v_with_runway(
        self, prompt: str, video_uri: str, ratio: Optional[str]
    ) -> Dict[str, Any]:
        """Call the Runway SDK video_to_video endpoint (model='aleph2').

        Aleph derives output length from the input video, so no `duration` is sent.
        """
        try:
            from runwayml import RunwayML, TaskFailedError
        except ImportError:
            raise ImportError("runwayml package not installed. Run: pip install runwayml")

        def _sync_generate():
            client = RunwayML()
            params = {
                "model": self.model,          # "aleph2"
                "video_uri": video_uri,
                "prompt_text": prompt,
            }
            if ratio:
                params["ratio"] = ratio
            try:
                task = client.video_to_video.create(**params).wait_for_task_output()
                video_url = None
                if getattr(task, "output", None):
                    video_url = task.output[0] if isinstance(task.output, list) else task.output
                return {
                    "task_id": getattr(task, "id", "unknown"),
                    "video_url": video_url,
                    "status": "success",
                }
            except TaskFailedError as e:
                logger.error(f"Runway Aleph v2v task failed: {e.task_details}")
                raise Exception(f"Runway Aleph v2v generation failed: {e.task_details}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate)

    async def _upload_file(self, file_path: Union[str, Path]) -> str:
        """Upload any file (image or video) via Runway's ephemeral upload."""
        return await self._upload_image(file_path)

    async def _upload_image(self, image_path: Union[str, Path]) -> str:
        """
        Upload image using Runway's ephemeral upload feature.
        
        Returns a runway:// URI that's valid for 24 hours and works directly
        with the Runway API - no external hosting (S3/CDN) required.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            from runwayml import RunwayML
        except ImportError:
            raise ImportError("runwayml package not installed. Run: pip install runwayml")
        
        # Run in thread pool since Runway SDK is synchronous
        def _sync_upload():
            client = RunwayML()
            
            # Use Runway's ephemeral upload - returns UploadCreateEphemeralResponse object
            # Valid for 24 hours, supports files up to 200 MB
            with open(path, "rb") as f:
                response = client.uploads.create_ephemeral(file=f)
            
            # Extract the URI string from the response object
            uri = response.uri
            logger.info(f"Uploaded image to Runway ephemeral storage: {uri}")
            return uri
        
        try:
            loop = asyncio.get_event_loop()
            image_uri = await loop.run_in_executor(None, _sync_upload)
            return image_uri
        except Exception as e:
            logger.error(f"Failed to upload image to Runway: {e}")
            raise Exception(f"Runway ephemeral upload failed: {e}")
    
    async def _generate_with_runway(
        self, 
        prompt: str, 
        image_url: str, 
        duration: int, 
        ratio: str
    ) -> Dict[str, Any]:
        """Generate video using Runway SDK."""
        try:
            from runwayml import RunwayML, TaskFailedError
        except ImportError:
            raise ImportError("runwayml package not installed. Run: pip install runwayml")
        
        # Run in thread pool since Runway SDK is synchronous
        def _sync_generate():
            client = RunwayML()
            
            try:
                task = client.image_to_video.create(
                    model=self.model,
                    prompt_image=image_url,
                    prompt_text=prompt,
                    ratio=ratio,
                    duration=duration
                ).wait_for_task_output()
                
                # Handle case where task.output is a list instead of string
                video_url = None
                if hasattr(task, 'output') and task.output:
                    if isinstance(task.output, list):
                        video_url = task.output[0] if task.output else None
                    else:
                        video_url = task.output
                
                return {
                    "task_id": task.id if hasattr(task, 'id') else 'unknown',
                    "video_url": video_url,
                    "status": "success"
                }
                
            except TaskFailedError as e:
                logger.error(f"Runway task failed: {e.task_details}")
                raise Exception(f"Runway generation failed: {e.task_details}")
            except Exception as e:
                logger.error(f"Runway SDK error: {e}")
                raise Exception(f"Runway generation error: {e}")
        
        # Run synchronous Runway call in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate)
    
    async def _download_video(self, video_url: str, output_path: Path) -> Path:
        """Download video from URL to local file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import httpx
        async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout for download
            response = await client.get(video_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download video: {response.status_code}")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        return output_path


# ========================================
# WRAPPER CLASS
# ========================================

class RunwayWrapper(ModelWrapper):
    """
    VBVR-InferKit wrapper for RunwayService to match the standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """Initialize Runway wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create RunwayService instance
        self.runway_service = RunwayService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        ratio: Optional[str] = None,
        video_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Runway (synchronous wrapper).

        Routing:
        - if `video_path` is provided (text + video -> video), use the Aleph
          video-to-video endpoint and ignore `image_path`.
        - otherwise, use the image-to-video endpoint (`image_path` + text).

        Args:
            image_path: Path to input image (first frame) for image-to-video.
            text_prompt: Text prompt for video generation.
            duration: Video duration in seconds (5 or 10 depending on model).
            output_filename: Optional output filename.
            ratio: Video aspect ratio (model-specific).
            video_path: Path to input video for video-to-video (Aleph). When set,
                the wrapper runs v2v instead of i2v.
            **kwargs: Additional parameters.

        Returns:
            Dictionary with the 8 standardized result fields.
        """
        start_time = time.time()

        # Convert duration to int (Runway expects int)
        duration_int = int(duration)

        # Generate output path
        if not output_filename:
            output_filename = "video.mp4"

        output_path = self.output_dir / output_filename

        is_v2v = video_path is not None
        try:
            if is_v2v:
                result = asyncio.run(
                    self.runway_service.generate_video_to_video(
                        prompt=text_prompt,
                        video_path=str(video_path),
                        duration=duration_int,
                        ratio=ratio,
                        output_path=output_path,
                    )
                )
            else:
                result = asyncio.run(
                    self.runway_service.generate_video(
                        prompt=text_prompt,
                        image_path=str(image_path),
                        duration=duration_int,
                        ratio=ratio,
                        output_path=output_path,
                    )
                )
        except Exception as e:
            logger.error(f"Runway generation failed: {e}")
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "generation_id": None,
                "model": self.model,
                "status": "failed",
                "metadata": {
                    "prompt": text_prompt,
                    "modality": "v2v" if is_v2v else "i2v",
                    "image_path": None if is_v2v else str(image_path),
                    "video_path_input": str(video_path) if is_v2v else None,
                },
            }

        duration_taken = time.time() - start_time

        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": result.get("task_id", 'unknown'),
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "modality": "v2v" if is_v2v else "i2v",
                "image_path": None if is_v2v else str(image_path),
                "video_path_input": str(video_path) if is_v2v else None,
                "video_url": result.get("video_url"),
                "duration": duration_int,
                "ratio": result.get("ratio"),
                "runway_result": result
            }
        }
