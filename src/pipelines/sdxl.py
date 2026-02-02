"""
Stable Diffusion XL pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional, List

import torch
from PIL import Image

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from .latent_mixins import SDLatentMixin
from ..schemas.models import ModelType

# Lazy imports - diffusers is only imported when actually needed
# This avoids loading all diffusers pipelines at startup

logger = logging.getLogger(__name__)


class SDXLPipeline(SDLatentMixin, BasePipeline):
    """
    Stable Diffusion XL pipeline.

    Supports:
    - output_type="pil" (default): Returns PIL Image
    - output_type="latent": Returns raw latent tensor
    - latents input: For img2img-style refinement with denoise_strength
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.SDXL,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        resolution_divisor=8,
        default_steps=30,
        default_guidance_scale=5.0,
        supports_negative_prompt=True,
        lora_merge_method=LoraMergeMethod.FUSE,
    )

    def _load_pipeline(self):
        """Load SDXL pipeline."""
        # Lazy import to avoid loading all diffusers pipelines at startup
        from diffusers import StableDiffusionXLPipeline, DDPMScheduler

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
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
        control_images: Optional[List[Image.Image]] = None,
        num_frames: int = 1,
        fps: int = 16,
        output_type: str = "pil",
        latents: Optional[torch.Tensor] = None,
        denoise_strength: float = 1.0,
    ) -> Dict[str, Any]:
        """Run SDXL inference with latent support."""
        # Handle latent input for img2img-style refinement
        if latents is not None:
            if denoise_strength <= 0:
                # No-op refinement: return input latents directly if requested
                if output_type == "latent":
                    return {"latents": latents}
                if hasattr(self, "decode_latent_to_image"):
                    return {"image": self.decode_latent_to_image(latents)}

            # Add noise to latents based on denoise_strength
            latents = latents.to(self.pipe.device, dtype=self.dtype)

            # Calculate the starting timestep based on denoise_strength
            init_timestep = min(int(num_inference_steps * denoise_strength), num_inference_steps)
            if init_timestep == 0:
                # No effective denoising steps -> return input directly
                if output_type == "latent":
                    return {"latents": latents}
                if hasattr(self, "decode_latent_to_image"):
                    return {"image": self.decode_latent_to_image(latents)}

            # Get timesteps
            self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.pipe.device)
            timesteps = self.pipe.scheduler.timesteps

            # Add noise to input latents
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start:]

            # Generate noise on CPU then move to device (generator compatibility)
            noise = torch.randn(
                latents.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=latents.device, dtype=latents.dtype)
            latents = self.pipe.scheduler.add_noise(latents, noise, timesteps[:1])

        # Determine output type for diffusers
        diffusers_output_type = "latent" if output_type == "latent" else "pil"

        # Build call kwargs
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": diffusers_output_type,
        }

        if latents is not None:
            call_kwargs["latents"] = latents
            # Pass explicit timesteps so denoise_strength takes effect (diffusers resets timesteps internally).
            call_kwargs["timesteps"] = timesteps
            # num_inference_steps already set above

        # Comfy-native progress + interrupt (no-op unless an observer is installed).
        self._inject_diffusers_callback_kwargs(call_kwargs, total_steps=num_inference_steps)

        result = self.pipe(**call_kwargs)

        if output_type == "latent":
            # diffusers returns raw latents when output_type="latent"
            return {"latents": result.images}
        else:
            return {"image": result.images[0]}
