"""
Z-Image De-Turbo pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)

# Scheduler config from ai-toolkit z_image.py
SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


class ZImageDeturboPipeline(BasePipeline):
    """
    Z-Image De-Turbo pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.ZIMAGE_DETURBO,
        base_model="Tongyi-MAI/Z-Image-Turbo",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=3.0,
        transformer_model="ostris/Z-Image-De-Turbo",
    )

    def _load_pipeline(self):
        """Load Z-Image De-Turbo pipeline.

        Components are loaded separately to avoid meta tensor issues:
        - transformer: from ostris/Z-Image-De-Turbo
        - text_encoder/vae/tokenizer: from Tongyi-MAI/Z-Image-Turbo
        """
        try:
            from diffusers import ZImagePipeline
            from diffusers.models.transformers import ZImageTransformer2DModel
            from transformers import AutoTokenizer, Qwen3ForCausalLM
        except ImportError as e:
            raise ImportError(
                f"ZImagePipeline requires latest diffusers and transformers. "
                f"Please upgrade: pip install diffusers transformers --upgrade. Error: {e}"
            )

        # Load transformer from De-Turbo model (MUST use ZImageTransformer2DModel)
        logger.info(f"Loading De-Turbo transformer: {self.CONFIG.transformer_model}")
        transformer = ZImageTransformer2DModel.from_pretrained(
            self.CONFIG.transformer_model,
            subfolder="transformer",
            torch_dtype=self.dtype,
            token=self.hf_token,
        )

        # Load text_encoder, tokenizer, vae from base model
        logger.info(f"Loading text_encoder/vae/tokenizer: {self.CONFIG.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.CONFIG.base_model,
            subfolder="tokenizer",
            token=self.hf_token,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self.CONFIG.base_model,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            token=self.hf_token,
        )
        vae = AutoencoderKL.from_pretrained(
            self.CONFIG.base_model,
            subfolder="vae",
            torch_dtype=self.dtype,
            token=self.hf_token,
        )

        # Create scheduler with ai-toolkit settings
        scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG)

        # Assemble pipeline with all components
        logger.info("Assembling ZImagePipeline")
        self.pipe = ZImagePipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            transformer=transformer,
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
        """Run Z-Image De-Turbo inference."""
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
