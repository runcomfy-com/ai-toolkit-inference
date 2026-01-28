"""
FLUX Kontext pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image
from diffusers import FluxKontextPipeline as DiffusersFluxKontextPipeline

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class FluxKontextPipeline(BasePipeline):
    """
    FLUX Kontext pipeline for image editing.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLUX_KONTEXT,
        base_model="black-forest-labs/FLUX.1-Kontext-dev",
        resolution_divisor=16,  # Aligned with training: get_bucket_divisibility() = 16
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
    )

    def _load_pipeline(self):
        """Load FLUX Kontext pipeline."""
        self.pipe = DiffusersFluxKontextPipeline.from_pretrained(
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
        """Run FLUX Kontext inference."""
        if control_image is None:
            raise ValueError("FLUX Kontext requires a control image")

        # Resize control image to match target dimensions
        # Use BILINEAR to align with training (extensions_built_in/diffusion_models/flux_kontext)
        control_image = control_image.resize((width, height), Image.BILINEAR)

        result = self.pipe(
            image=control_image,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            negative_prompt=negative_prompt,
            # Aligned with training: use actual image area for dynamic shifting calculation
            max_area=height * width,
            _auto_resize=False,
        )

        return {"image": result.images[0]}
