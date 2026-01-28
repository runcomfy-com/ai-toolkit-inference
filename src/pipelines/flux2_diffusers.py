"""
FLUX.2-dev pipeline implementation using diffusers.
"""

import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image
from diffusers import Flux2Pipeline as DiffusersFlux2Pipeline

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class Flux2DiffusersPipeline(BasePipeline):
    """
    FLUX.2-dev pipeline (diffusers).
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLUX2_DIFFUSERS,
        base_model="black-forest-labs/FLUX.2-dev",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        supports_negative_prompt=False,
        lora_merge_method=LoraMergeMethod.SET_ADAPTERS,  # merge LoRA into weights for speed
    )

    def _load_pipeline(self):
        """Load FLUX.2-dev pipeline via diffusers."""
        self.pipe = DiffusersFlux2Pipeline.from_pretrained(
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
        num_frames: int = 1,
        fps: int = 16,
    ) -> Dict[str, Any]:
        """Run FLUX.2-dev inference via diffusers."""
        if control_image or control_images:
            logger.warning("FLUX.2 diffusers pipeline does not support control images, ignoring")

        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            negative_prompt=negative_prompt or "",
        )

        return {"image": result.images[0]}
