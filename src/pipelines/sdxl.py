"""
Stable Diffusion XL pipeline implementation.
"""

import logging
from typing import Dict, Any, Optional, List

import torch
from PIL import Image

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType

# Lazy imports - diffusers is only imported when actually needed
# This avoids loading all diffusers pipelines at startup

logger = logging.getLogger(__name__)


class SDXLPipeline(BasePipeline):
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

    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to latent space using the VAE."""
        import numpy as np

        # With CPU offload, VAE params are on CPU but hooks move them to GPU on forward
        target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Convert PIL to tensor [B, C, H, W] in [-1, 1]
        img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor * 2.0 - 1.0).to(target_device, dtype=self.dtype)

        # Move VAE to GPU explicitly if using CPU offload
        if self.enable_cpu_offload:
            self.pipe.vae.to(target_device)

        # Encode to latent
        with torch.no_grad():
            latent = self.pipe.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * self.pipe.vae.config.scaling_factor

        return latent

    def encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Batch encode images [B,C,H,W] in [-1,1] to latents [B,C,H/8,W/8].
        
        This is more efficient than looping encode_image_to_latent for batches.
        """
        target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        images = images.to(target_device, dtype=self.dtype)

        if self.enable_cpu_offload:
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            latents = self.pipe.vae.encode(images).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor

        return latents

    def decode_latent_to_image(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to PIL image using the VAE."""
        import numpy as np

        # With CPU offload, VAE params are on CPU but hooks move them to GPU on forward
        # We need to ensure latents are on the target compute device (cuda:0)
        target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        latents = latents.to(target_device, dtype=self.dtype)
        latents = latents / self.pipe.vae.config.scaling_factor

        # Move VAE to GPU explicitly if using CPU offload
        if self.enable_cpu_offload:
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.float().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """Batch decode latents [B,C,H/8,W/8] to images [B,C,H,W] in [0,1].
        
        This is more efficient than looping decode_latent_to_image for batches.
        Returns tensor in [0,1] range with shape [B,C,H,W].
        """
        target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        latents = latents.to(target_device, dtype=self.dtype)
        latents = latents / self.pipe.vae.config.scaling_factor

        if self.enable_cpu_offload:
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            images = self.pipe.vae.decode(latents).sample

        # Convert from [-1,1] to [0,1]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

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
            # Add noise to latents based on denoise_strength
            latents = latents.to(self.pipe.device, dtype=self.dtype)

            # Calculate the starting timestep based on denoise_strength
            init_timestep = min(int(num_inference_steps * denoise_strength), num_inference_steps)

            # Get timesteps
            self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.pipe.device)
            timesteps = self.pipe.scheduler.timesteps

            # Add noise to input latents
            if denoise_strength < 1.0:
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
            # Adjust steps for img2img
            if denoise_strength < 1.0:
                call_kwargs["num_inference_steps"] = num_inference_steps

        result = self.pipe(**call_kwargs)

        if output_type == "latent":
            # diffusers returns raw latents when output_type="latent"
            return {"latents": result.images}
        else:
            return {"image": result.images[0]}
