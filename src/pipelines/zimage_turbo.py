"""
Z-Image Turbo pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class ZImageTurboPipeline(BasePipeline):
    """
    Z-Image Turbo pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.ZIMAGE_TURBO,
        base_model="Tongyi-MAI/Z-Image-Turbo",
        resolution_divisor=32,
        default_steps=8,
        default_guidance_scale=1.0,
    )

    def _load_pipeline(self):
        """Load Z-Image Turbo pipeline."""
        try:
            from diffusers import ZImagePipeline

            self.pipe = ZImagePipeline.from_pretrained(
                self.CONFIG.base_model,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        except ImportError:
            # Fallback to FluxPipeline if ZImagePipeline not available
            from diffusers import FluxPipeline

            logger.warning("ZImagePipeline not available, using FluxPipeline")
            self.pipe = FluxPipeline.from_pretrained(
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
        """Run Z-Image Turbo inference."""
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            negative_prompt=negative_prompt,
        )

        return {"image": result.images[0]}
