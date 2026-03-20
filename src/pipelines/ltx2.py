"""
LTX-2 pipeline implementation.

Supports both T2V (text-to-video) and I2V (image-to-video) modes.
- T2V: No control_image provided → uses LTX2Pipeline
- I2V: control_image provided → uses LTX2ImageToVideoPipeline

"""

import os
import sys
import tempfile
import logging
import time
from typing import Dict, Any, Optional

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType
from ..config import settings

logger = logging.getLogger(__name__)


class LTX2Pipeline(BasePipeline):
    """
    LTX-2 pipeline for video generation with audio.

    Key features:
    - Unified model type for both T2V and I2V (controlled by control_image parameter)
    - Uses official diffusers LTX2Pipeline / LTX2ImageToVideoPipeline
    - LoRA conversion from ai-toolkit format to diffusers format
    - Audio generation included
    - Output: frames + audio (saved as MP4 by executor)
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.LTX2,
        base_model="Lightricks/LTX-2",
        resolution_divisor=32,  # Width/height must be divisible by 32
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=False,  # Optional for I2V
        is_video_model=True,
        default_num_frames=41,  # Must be 8N+1: 1, 9, 17, 25, 33, 41...
        default_fps=24,
        enable_cpu_offload=True,  # Large model, benefits from offload
        lora_merge_method=LoraMergeMethod.SET_ADAPTERS,  # Hotswap mode (recommended)
        enable_xformers=True,  # Enable xformers memory efficient attention
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._i2v_pipe = None  # Lazy-created for I2V mode
        self.timings = {}  # Store pipeline timing details
        self._lora_fused = False  # Track if LoRA is currently fused

    def _load_pipeline(self):
        """Load LTX-2 pipeline using from_pretrained.

        Note: Pre-download with ignore_patterns is handled by PipelineManager
        to skip large single-file checkpoints (~170GB) that are only for ComfyUI.
        """
        from diffusers import LTX2Pipeline as DiffusersLTX2Pipeline

        logger.info(f"Loading LTX-2 pipeline from: {self.CONFIG.base_model}")

        self.pipe = DiffusersLTX2Pipeline.from_pretrained(
            self.CONFIG.base_model,
            torch_dtype=self.dtype,
            token=self.hf_token,
        )

        logger.info(f"LTX-2 pipeline loaded. Scheduler: {type(self.pipe.scheduler).__name__}")

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """
        Load LoRA for LTX-2.

        ai-toolkit saves LoRA in original format, needs conversion to diffusers format.

        Args:
            lora_paths: List of LoRA file paths (uses first one)
            lora_scale: LoRA weight scale
        """
        if not lora_paths:
            logger.warning("No LoRA paths provided")
            return

        lora_path = lora_paths[0]
        if isinstance(lora_path, dict):
            # Handle MoE format (not used for LTX-2, but handle gracefully)
            lora_path = lora_path.get("high") or lora_path.get("low")

        if not lora_path or not os.path.exists(lora_path):
            logger.warning(f"LoRA file not found: {lora_path}")
            return

        logger.info(f"Loading LoRA: {lora_path}")

        # Load raw LoRA
        t0 = time.perf_counter()
        lora_sd = load_file(lora_path)
        load_time = time.perf_counter() - t0
        logger.info(f"[TIMING] lora_load_file: {load_time:.3f}s (keys={len(lora_sd)})")

        # Convert to diffusers format
        t0 = time.perf_counter()
        converted_sd = self._convert_lora_to_diffusers(lora_sd)
        convert_time = time.perf_counter() - t0
        logger.info(f"[TIMING] lora_convert_to_diffusers: {convert_time:.3f}s (keys={len(converted_sd)})")

        # Save to temp file and load via diffusers
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            temp_path = f.name

        try:
            t0 = time.perf_counter()
            save_file(converted_sd, temp_path)
            save_time = time.perf_counter() - t0
            logger.info(f"[TIMING] lora_save_temp: {save_time:.3f}s")

            t0 = time.perf_counter()
            self.pipe.load_lora_weights(
                os.path.dirname(temp_path),
                weight_name=os.path.basename(temp_path),
                adapter_name="lora",
            )
            load_weights_time = time.perf_counter() - t0
            logger.info(f"[TIMING] lora_load_weights: {load_weights_time:.3f}s")

            # Check merge method from config
            merge_method = self.CONFIG.lora_merge_method
            if merge_method == LoraMergeMethod.FUSE:
                # Fuse mode: merge LoRA into base weights
                t0 = time.perf_counter()
                self.pipe.fuse_lora(adapter_names=["lora"], lora_scale=lora_scale)
                self.pipe.unload_lora_weights()
                fuse_time = time.perf_counter() - t0
                self._lora_fused = True
                self._num_loras_fused = 1
                logger.info(f"[TIMING] lora_fuse: {fuse_time:.3f}s")
                logger.info(f"LoRA loaded (fuse mode, scale={lora_scale})")
            else:
                # SET_ADAPTERS mode: keep as adapter
                self._lora_fused = False
                self._num_loras_fused = 0
                logger.info(f"LoRA loaded (adapter mode, scale=1.0)")

            self.lora_loaded = True
            self._current_lora_scale = lora_scale
            self._current_lora_paths = [lora_path]

            total_lora_time = load_time + convert_time + save_time + load_weights_time

            # Store timing data for API response
            self.timings.update(
                {
                    "lora_load_file": load_time,
                    "lora_convert_to_diffusers": convert_time,
                    "lora_save_temp": save_time,
                    "lora_load_weights": load_weights_time,
                    "lora_total": total_lora_time,
                }
            )

            logger.info(
                f"[TIMING] lora_total: {total_lora_time:.3f}s | "
                f"load={load_time:.3f}s, convert={convert_time:.3f}s, save={save_time:.3f}s, "
                f"load_weights={load_weights_time:.3f}s"
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _convert_lora_to_diffusers(self, lora_sd: dict) -> dict:
        """
        Convert ai-toolkit LoRA to diffusers format.

        Uses conversion logic from ai-toolkit's convert_ltx2_to_diffusers.py
        """
        # Try to import conversion function from ai-toolkit (configurable via AI_TOOLKIT_PATH env var)
        ai_toolkit_path = settings.ai_toolkit_path
        if os.path.exists(ai_toolkit_path) and ai_toolkit_path not in sys.path:
            sys.path.insert(0, ai_toolkit_path)

        try:
            from extensions_built_in.diffusion_models.ltx2.convert_ltx2_to_diffusers import (
                convert_lora_original_to_diffusers,
            )

            converted_sd = convert_lora_original_to_diffusers(lora_sd)
        except ImportError:
            logger.warning("Could not import convert_lora_original_to_diffusers, using fallback")
            converted_sd = lora_sd

        # Replace prefix: diffusion_model. -> transformer.
        final_sd = {}
        for k, v in converted_sd.items():
            new_key = k.replace("diffusion_model.", "transformer.")
            final_sd[new_key] = v

        return final_sd

    def set_lora_scale(self, scale: float) -> bool:
        """
        Set LoRA scale dynamically.

        - SET_ADAPTERS mode: Uses low-level PEFT API to set scale directly
        - FUSE mode: Requires unfuse -> reload -> fuse(new_scale)

        Args:
            scale: New LoRA scale value

        Returns:
            True if scale changed, False otherwise
        """
        if not self.lora_loaded:
            return False

        if scale == self._current_lora_scale:
            return True

        merge_method = self.CONFIG.lora_merge_method

        if merge_method == LoraMergeMethod.FUSE:
            # Fuse mode: need unfuse -> reload -> fuse(new_scale)
            logger.info(f"Changing fused LoRA scale: unfuse -> reload -> fuse(scale={scale})")
            try:
                if self._lora_fused:
                    self.pipe.unfuse_lora()
                    self._lora_fused = False
                # Reload with new scale (will fuse again)
                self._load_lora(self._current_lora_paths, scale)
                return True
            except Exception as e:
                logger.warning(f"Failed to change LoRA scale: {e}")
                return False
        else:
            # SET_ADAPTERS mode: use low-level PEFT API to set scale
            try:
                self._set_lora_scale_direct(scale)
                self._current_lora_scale = scale
                logger.info(f"LTX-2 adapter mode: set scale to {scale}")
                return True
            except Exception as e:
                logger.warning(f"Failed to set LoRA scale: {e}")
                return False

    def switch_lora(self, lora_paths: list, lora_scale: float = 1.0) -> bool:
        """
        Switch to a new LoRA.

        Strategy depends on lora_merge_method:
        - SET_ADAPTERS: Uses hotswap for in-place replacement (fast)
        - FUSE: Uses unfuse -> load -> fuse (slower but supports scale)

        Args:
            lora_paths: New LoRA paths to load
            lora_scale: LoRA strength (0.0 to 2.0)

        Returns:
            True if successfully switched
        """
        if self.pipe is None:
            logger.warning("Pipeline not loaded, cannot switch LoRA")
            return False

        if not lora_paths:
            logger.warning("No LoRA paths provided")
            return False

        merge_method = self.CONFIG.lora_merge_method

        if merge_method == LoraMergeMethod.FUSE:
            # Fuse mode: unfuse -> load -> fuse
            if self._lora_fused:
                logger.info("Unfusing current LoRA before loading new one")
                try:
                    self.pipe.unfuse_lora()
                    self._lora_fused = False
                except Exception as e:
                    logger.warning(f"Failed to unfuse LoRA: {e}")
                    return False
            # Load and fuse new LoRA
            self._load_lora(lora_paths, lora_scale)
            return True
        else:
            # SET_ADAPTERS mode: use hotswap for direct in-place replacement
            self._load_lora_hotswap(lora_paths, lora_scale)
            return True

    def _load_lora_hotswap(self, lora_paths: list, lora_scale: float = 1.0):
        """
        Load LoRA using hotswap=True for in-place replacement.

        Args:
            lora_paths: List of LoRA file paths
            lora_scale: LoRA weight scale (note: scale only works if fuse_lora is used)
        """
        lora_path = lora_paths[0]
        if isinstance(lora_path, dict):
            lora_path = lora_path.get("high") or lora_path.get("low")

        if not lora_path or not os.path.exists(lora_path):
            logger.warning(f"LoRA file not found: {lora_path}")
            return

        logger.info(f"Hotswap LoRA: {lora_path}")

        # Load and convert LoRA
        t0 = time.perf_counter()
        lora_sd = load_file(lora_path)
        load_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        converted_sd = self._convert_lora_to_diffusers(lora_sd)
        convert_time = time.perf_counter() - t0

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            temp_path = f.name

        try:
            t0 = time.perf_counter()
            save_file(converted_sd, temp_path)
            save_time = time.perf_counter() - t0

            # Load with hotswap=True for in-place replacement
            t0 = time.perf_counter()
            self.pipe.load_lora_weights(
                os.path.dirname(temp_path),
                weight_name=os.path.basename(temp_path),
                adapter_name="lora",
                hotswap=True,  # Key: in-place replacement
            )
            hotswap_time = time.perf_counter() - t0

            total_time = load_time + convert_time + save_time + hotswap_time
            self._current_lora_paths = [lora_path]

            logger.info(
                f"[TIMING] hotswap_total: {total_time:.3f}s | "
                f"load={load_time:.3f}s, convert={convert_time:.3f}s, "
                f"save={save_time:.3f}s, hotswap={hotswap_time:.3f}s"
            )
            logger.info(f"LoRA hotswapped successfully")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _get_i2v_pipeline(self):
        """Get or create I2V pipeline from T2V components."""
        if self._i2v_pipe is None:
            from diffusers import LTX2ImageToVideoPipeline

            logger.info("Creating I2V pipeline from T2V components")
            self._i2v_pipe = LTX2ImageToVideoPipeline(
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                audio_vae=self.pipe.audio_vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                connectors=self.pipe.connectors,
                transformer=self.pipe.transformer,
                vocoder=self.pipe.vocoder,
            )

        return self._i2v_pipe

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        control_image: Optional[Image.Image] = None,
        control_images: Optional[list] = None,
        num_frames: int = 41,
        fps: int = 24,
    ) -> Dict[str, Any]:
        """
        Run LTX-2 inference.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Output width (must be divisible by 32)
            height: Output height (must be divisible by 32)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            generator: Random generator
            control_image: Control image for I2V mode (optional)
            control_images: Not used for LTX-2
            num_frames: Number of frames (auto-adjusted to 8N+1)
            fps: Frames per second

        Returns:
            Dict with frames, fps, audio, and audio_sample_rate
        """
        # Note: num_frames must be 8N+1 (1, 9, 17, 25...) - pipeline handles adjustment internally

        # Common kwargs for both pipelines
        # Use output_type="np" to get numpy array directly (aligned with ai-toolkit)
        common_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "frame_rate": fps,
            "output_type": "np",  # numpy array [batch, frames, channels, height, width]
            "return_dict": False,
        }

        # Select pipeline based on control_image
        if control_image is not None:
            # I2V mode
            logger.info("Running I2V inference (control_image provided)")
            pipe = self._get_i2v_pipeline()

            # Resize control image to match output dimensions
            control_image = control_image.resize((width, height), Image.LANCZOS)
            common_kwargs["image"] = control_image
        else:
            # T2V mode
            logger.info("Running T2V inference")
            pipe = self.pipe

        # Run inference
        # Comfy-native progress + interrupt (no-op unless an observer is installed).
        self._inject_diffusers_callback_kwargs(common_kwargs, total_steps=num_inference_steps, pipe=pipe)
        video, audio = pipe(**common_kwargs)

        # video is numpy array [batch, frames, channels, height, width]
        # Convert to torch tensor and scale to uint8 (aligned with ai-toolkit)
        video = (video * 255).round().astype("uint8")
        video_tensor = torch.from_numpy(video[0])  # [frames, channels, height, width]

        # Get audio sample rate from vocoder config
        audio_sample_rate = 24000  # Default
        if hasattr(pipe, "vocoder") and hasattr(pipe.vocoder, "config"):
            audio_sample_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

        # Prepare audio tensor
        audio_tensor = None
        if audio is not None:
            if isinstance(audio, list):
                audio_tensor = audio[0] if len(audio) > 0 else None
            else:
                audio_tensor = audio
            if audio_tensor is not None:
                audio_tensor = audio_tensor.float().cpu()

        # Return video tensor and audio for encode_video
        # video_tensor is [T, C, H, W] uint8 torch tensor
        return {
            "video_tensor": video_tensor,
            "fps": fps,
            "audio": audio_tensor,
            "audio_sample_rate": audio_sample_rate,
        }

    def unload(self):
        """Unload the pipeline to free memory."""
        if self._i2v_pipe is not None:
            del self._i2v_pipe
            self._i2v_pipe = None
        super().unload()


class LTX23Pipeline(LTX2Pipeline):
    """
    LTX-2.3 pipeline for video generation with audio.

    Inherits all behavior from LTX2Pipeline. The only difference is the base model:
    dg845/LTX-2.3-Diffusers ships LTX2VocoderWithBWE in its model_index.json,
    so diffusers from_pretrained() automatically loads the correct vocoder.

    Supports resolution="High" mode which runs inference at half resolution
    then applies 2x spatial upsampling via LTX2LatentUpsamplerModel.
    """

    UPSAMPLER_REPO = "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"

    CONFIG = PipelineConfig(
        model_type=ModelType.LTX2_3,
        base_model="dg845/LTX-2.3-Diffusers",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=False,
        is_video_model=True,
        default_num_frames=41,
        default_fps=24,
        enable_cpu_offload=True,
        lora_merge_method=LoraMergeMethod.SET_ADAPTERS,
        enable_xformers=True,
        supported_resolution_modes=["Default", "High"],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._upsample_pipe = None

    def _get_upsample_pipeline(self):
        """Get or create the latent upsampler pipeline."""
        if self._upsample_pipe is not None:
            return self._upsample_pipe

        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

        logger.info(f"Loading latent upsampler from: {self.UPSAMPLER_REPO}")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.UPSAMPLER_REPO,
            subfolder="latent_upsampler",
            torch_dtype=self.dtype,
            token=self.hf_token,
        )

        self._upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipe.vae,
            latent_upsampler=latent_upsampler,
        )
        self._upsample_pipe.vae.enable_tiling()

        # Apply offload strategy consistent with the main pipeline
        if self.offload_mode == "sequential":
            if hasattr(self._upsample_pipe, "enable_sequential_cpu_offload"):
                self._upsample_pipe.enable_sequential_cpu_offload()
            else:
                self._upsample_pipe.enable_model_cpu_offload()
        elif self.offload_mode == "model":
            self._upsample_pipe.enable_model_cpu_offload()
        else:
            self._upsample_pipe.to(self.device)

        logger.info("Latent upsampler loaded")
        return self._upsample_pipe

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        control_image: Optional[Image.Image] = None,
        control_images: Optional[list] = None,
        num_frames: int = 41,
        fps: int = 24,
        resolution: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run LTX-2.3 inference with optional 2x upscale."""
        if resolution != "High":
            return super()._run_inference(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                control_image=control_image,
                control_images=control_images,
                num_frames=num_frames,
                fps=fps,
            )

        # High resolution mode: stage 1 at half res → 2x upsample.
        # 2x upscale requires dimensions divisible by 64 (= 32 * 2) so that
        # the half-resolution stage lands on a clean 32px grid.
        # Auto-snap to nearest 64-grid (round to nearest, prefer larger).
        divisor = self.CONFIG.resolution_divisor
        grid = divisor * 2  # 64
        snapped_w = ((width + grid // 2) // grid) * grid
        snapped_h = ((height + grid // 2) // grid) * grid
        if snapped_w != width or snapped_h != height:
            logger.warning(
                f"High mode: snapping {width}x{height} to {snapped_w}x{snapped_h} "
                f"(must be divisible by {grid} for 2x upscale)"
            )
            width = snapped_w
            height = snapped_h
        half_w = width // 2
        half_h = height // 2
        logger.info(
            f"High resolution mode: stage 1 at {half_w}x{half_h}, "
            f"upsample to {width}x{height}"
        )

        # Stage 1: run at half resolution with latent output to avoid decode+re-encode
        t0 = time.perf_counter()
        common_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "height": half_h,
            "width": half_w,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "frame_rate": fps,
            "output_type": "latent",
            "return_dict": False,
        }

        if control_image is not None:
            pipe = self._get_i2v_pipeline()
            control_image = control_image.resize((half_w, half_h), Image.LANCZOS)
            common_kwargs["image"] = control_image
        else:
            pipe = self.pipe

        self._inject_diffusers_callback_kwargs(common_kwargs, total_steps=num_inference_steps, pipe=pipe)
        video_latents, audio_latents = pipe(**common_kwargs)
        stage1_time = time.perf_counter() - t0
        logger.info(f"[TIMING] stage1_inference: {stage1_time:.3f}s")

        # Decode audio from latents (audio_latents are raw, not decoded in latent mode)
        # Note: after pipe() returns, maybe_free_model_hooks() may have moved models to CPU.
        # Explicitly move audio components to the correct device before decoding.
        t0 = time.perf_counter()
        device = video_latents.device
        pipe.audio_vae.to(device)
        pipe.vocoder.to(device)
        audio_latents_decoded = audio_latents.to(device=device, dtype=pipe.audio_vae.dtype)
        generated_mel = pipe.audio_vae.decode(audio_latents_decoded, return_dict=False)[0]
        audio = pipe.vocoder(generated_mel)
        audio_decode_time = time.perf_counter() - t0
        logger.info(f"[TIMING] audio_decode: {audio_decode_time:.3f}s")

        audio_sample_rate = 24000
        if hasattr(pipe, "vocoder") and hasattr(pipe.vocoder, "config"):
            audio_sample_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

        audio_tensor = None
        if audio is not None:
            if isinstance(audio, list):
                audio_tensor = audio[0] if len(audio) > 0 else None
            else:
                audio_tensor = audio
            if audio_tensor is not None:
                audio_tensor = audio_tensor.float().cpu()

        # Stage 2: latent upsample (2x spatial)
        t0 = time.perf_counter()
        upsample_pipe = self._get_upsample_pipeline()

        # video_latents from stage 1 are already denormalized (diffusers denormalizes before output_type check)
        upsampled_video = upsample_pipe(
            latents=video_latents,
            latents_normalized=False,
            height=half_h,
            width=half_w,
            num_frames=num_frames,
            output_type="np",
            return_dict=False,
        )[0]
        upsample_time = time.perf_counter() - t0
        logger.info(f"[TIMING] latent_upsample: {upsample_time:.3f}s")

        logger.info(
            f"[TIMING] high_res_total: {stage1_time + audio_decode_time + upsample_time:.3f}s "
            f"(stage1={stage1_time:.3f}s, audio={audio_decode_time:.3f}s, upsample={upsample_time:.3f}s)"
        )

        # Convert to uint8 tensor
        upsampled_video = (upsampled_video * 255).round().astype("uint8")
        video_tensor = torch.from_numpy(upsampled_video[0])

        return {
            "video_tensor": video_tensor,
            "fps": fps,
            "audio": audio_tensor,
            "audio_sample_rate": audio_sample_rate,
        }

    def unload(self):
        """Unload the pipeline and upsampler to free memory."""
        if self._upsample_pipe is not None:
            del self._upsample_pipe
            self._upsample_pipe = None
        super().unload()
