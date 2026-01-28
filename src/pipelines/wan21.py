"""
Wan 2.1 T2V pipeline implementations.
"""

import os
import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image
from diffusers import WanPipeline

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class Wan21T2V14BPipeline(BasePipeline):
    """
    Wan 2.1 T2V 14B pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.WAN21_14B,
        base_model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        is_video_model=True,
        default_num_frames=41,
        default_fps=16,
    )

    def _load_pipeline(self):
        """Load Wan 2.1 T2V 14B pipeline."""
        self.pipe = WanPipeline.from_pretrained(
            self.CONFIG.base_model,
            torch_dtype=self.dtype,
            token=self.hf_token,
        )

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """Load single LoRA for Wan 2.1."""
        if not lora_paths:
            logger.warning("No LoRA paths provided")
            return

        lora_path = lora_paths[0]
        logger.info(f"Loading LoRA: {lora_path}")

        lora_dir = os.path.dirname(lora_path)
        lora_file = os.path.basename(lora_path)

        self.pipe.load_lora_weights(
            lora_dir,
            weight_name=lora_file,
            adapter_name="lora",
            local_files_only=True,
        )

        self.pipe.set_adapters(["lora"], adapter_weights=[lora_scale])
        self.lora_loaded = True
        logger.info(f"LoRA loaded with scale={lora_scale}")

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
        fps: int = 16,
    ) -> Dict[str, Any]:
        """Run Wan 2.1 T2V inference."""

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        # result.frames is a list of lists (batch of videos, each video is list of frames)
        frames = result.frames[0] if hasattr(result, "frames") else result.images

        return {"frames": frames, "fps": fps}


class Wan21T2V1BPipeline(Wan21T2V14BPipeline):
    """Wan 2.1 T2V 1B pipeline."""

    CONFIG = PipelineConfig(
        model_type=ModelType.WAN21_1B,
        base_model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        is_video_model=True,
        default_num_frames=41,
        default_fps=16,
    )


class Wan21I2V14BPipeline(Wan21T2V14BPipeline):
    """
    Wan 2.1 I2V 14B pipeline.

    Inherits from T2V to reuse _load_lora.
    Requires control image for image-to-video generation.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.WAN21_I2V_14B,
        base_model="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
        is_video_model=True,
        default_num_frames=41,
        default_fps=16,
    )

    def _load_pipeline(self):
        """Load Wan 2.1 I2V 14B pipeline."""
        from diffusers import WanImageToVideoPipeline

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.CONFIG.base_model,
            torch_dtype=self.dtype,
            token=self.hf_token,
        )

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
        fps: int = 16,
    ) -> Dict[str, Any]:
        """Run Wan 2.1 I2V inference."""
        if control_image is None:
            raise ValueError("Wan 2.1 I2V requires a control image")

        # Resize control image to match output dimensions (aligned with toolkit)
        control_image = control_image.resize((width, height), Image.LANCZOS)

        result = self.pipe(
            image=control_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        frames = result.frames[0] if hasattr(result, "frames") else result.images

        return {"frames": frames, "fps": fps}


class Wan21I2V14B480PPipeline(Wan21I2V14BPipeline):
    """Wan 2.1 I2V 14B 480P version."""

    CONFIG = PipelineConfig(
        model_type=ModelType.WAN21_I2V_14B_480P,
        base_model="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
        is_video_model=True,
        default_num_frames=41,
        default_fps=16,
    )
