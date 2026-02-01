"""
FLUX.1-dev pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image
from diffusers import FluxPipeline

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class FluxDevPipeline(BasePipeline):
    """
    FLUX.1-dev pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLUX,
        base_model="black-forest-labs/FLUX.1-dev",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
    )

    def _load_pipeline(self):
        """Load FLUX.1-dev pipeline."""
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
        """Run FLUX.1-dev inference."""
        call_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "negative_prompt": negative_prompt,
        }
        # Comfy-native progress + interrupt (no-op unless an observer is installed).
        self._inject_diffusers_callback_kwargs(call_kwargs, total_steps=num_inference_steps)
        result = self.pipe(**call_kwargs)

        return {"image": result.images[0]}
