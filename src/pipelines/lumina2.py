"""
Lumina2 pipeline implementation.

Uses diffusers' Lumina2Pipeline with Gemma2 text encoder.
"""

import gc
import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)

# Scheduler config - aligned with ai-toolkit lumina2_config from toolkit/sampler.py
LUMINA2_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 6.0,
    "shift_terminal": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class Lumina2Pipeline(BasePipeline):
    """
    Lumina2 text-to-image pipeline.

    Uses diffusers' Lumina2Pipeline and Lumina2Transformer2DModel with Gemma2 text encoder.

    Key alignment points:
    - Custom model loading from ai-toolkit patterns
    - LoRA via load_lora_weights + set_adapters (uses base class _load_lora)
    - Gemma2 text encoder
    - Resolution divisor: 32
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.LUMINA2,
        base_model="Alpha-VLLM/Lumina-Image-2.0",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = None
        self.tokenizer = None
        self.transformer = None

    def _load_pipeline(self):
        """Load Lumina2 pipeline using diffusers components."""
        try:
            from diffusers import (
                AutoencoderKL,
                FlowMatchEulerDiscreteScheduler,
                Lumina2Pipeline as DiffusersLumina2Pipeline,
                Lumina2Transformer2DModel,
            )
            from transformers import AutoTokenizer, AutoModel

            model_path = self.CONFIG.base_model

            # ========== Step 1: Load transformer ==========
            logger.info("[1/3] Loading Lumina2 transformer")
            transformer = Lumina2Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            transformer.to(self.device, dtype=self.dtype)

            # ========== Step 2: Load Gemma2 text encoder ==========
            logger.info("[2/3] Loading Gemma2 text encoder")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            text_encoder = AutoModel.from_pretrained(
                model_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            text_encoder.to(self.device, dtype=self.dtype)

            # ========== Step 3: Load scheduler, VAE and create pipeline ==========
            logger.info("[3/3] Loading scheduler and VAE")

            scheduler = FlowMatchEulerDiscreteScheduler(**LUMINA2_SCHEDULER_CONFIG)

            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            vae.to(self.device, dtype=self.dtype)

            # Create pipeline
            logger.info("Creating Lumina2 pipeline")
            self.pipe = DiffusersLumina2Pipeline(
                scheduler=scheduler,
                text_encoder=None,
                tokenizer=tokenizer,
                vae=vae,
                transformer=None,
            )
            self.pipe.text_encoder = text_encoder
            self.pipe.transformer = transformer

            # Store references
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self.transformer = transformer

            # Set eval mode and disable gradients
            transformer.requires_grad_(False)
            transformer.eval()
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            vae.requires_grad_(False)
            vae.eval()

            # Print GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

            logger.info("Lumina2 pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Lumina2 pipeline: {e}")
            raise

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
        """Run Lumina2 inference."""
        neg_prompt = negative_prompt if negative_prompt else ""

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "max_sequence_length": 256,
        }
        # Comfy-native progress + interrupt (no-op unless an observer is installed).
        self._inject_diffusers_callback_kwargs(call_kwargs, total_steps=num_inference_steps)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            result = self.pipe(**call_kwargs)

        return {"image": result.images[0]}

    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.lora_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Lumina2 pipeline unloaded")
