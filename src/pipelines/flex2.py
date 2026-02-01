"""
Flex.2 pipeline implementation.
"""

import os
import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, FlowMatchEulerDiscreteScheduler

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class Flex2Pipeline(BasePipeline):
    """
    Flex.2 pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLEX2,
        base_model="ostris/Flex.2-preview",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        lora_merge_method=LoraMergeMethod.FUSE,  # fuse_lora merges weights
    )

    def _load_pipeline(self):
        """Load Flex.2 pipeline."""
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.CONFIG.base_model,
            custom_pipeline=self.CONFIG.base_model,
            torch_dtype=self.dtype,
            token=self.hf_token,
        )

        # Configure scheduler (ai-toolkit settings)
        # Key: shift=3.0 and use_dynamic_shifting=True
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler(
            base_image_seq_len=256,
            base_shift=0.5,
            max_image_seq_len=4096,
            max_shift=1.15,
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
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
        """Run Flex.2 inference."""
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
