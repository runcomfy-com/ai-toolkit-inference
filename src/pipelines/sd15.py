"""
Stable Diffusion 1.5 pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class SD15Pipeline(BasePipeline):
    """
    Stable Diffusion 1.5 pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.SD15,
        base_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        resolution_divisor=8,
        default_steps=25,
        default_guidance_scale=6.0,
        supports_negative_prompt=True,
    )

    def _load_pipeline(self):
        """Load SD1.5 pipeline."""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.CONFIG.base_model,
            torch_dtype=self.dtype,
            token=self.hf_token,
            variant="fp16",
        )

        # Configure scheduler (aligned with ai-toolkit sampler.py sd_config)
        self.pipe.scheduler = DDPMScheduler.from_config(
            {
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "clip_sample": False,
                "num_train_timesteps": 1000,
                "prediction_type": "epsilon",
                "set_alpha_to_one": False,
                "steps_offset": 0,
                "timestep_spacing": "leading",
                "trained_betas": None,
            }
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
        num_frames: int = 1,
        fps: int = 16,
    ) -> Dict[str, Any]:
        """Run SD1.5 inference."""
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return {"image": result.images[0]}
