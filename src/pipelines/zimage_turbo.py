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
        except ImportError as e:
            raise ImportError(
                f"ZImagePipeline requires latest diffusers. "
                f"Please upgrade: pip install diffusers --upgrade. Error: {e}"
            )

        self.pipe = ZImagePipeline.from_pretrained(
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
