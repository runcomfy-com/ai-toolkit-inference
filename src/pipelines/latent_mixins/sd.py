"""
Latent helpers for SD/SDXL-style VAE pipelines.
"""

from typing import Optional

import torch
from PIL import Image

from .base import LatentContractMixin


class SDLatentMixin(LatentContractMixin):
    """
    Mixin for SD/SDXL latent encode/decode and metadata.
    """

    LATENT_NORM = "sd_scaling"
    LATENT_PACKED = False

    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to latent space using the VAE."""
        import numpy as np

        target_device = self._get_execution_device()

        # Convert PIL to tensor [B, C, H, W] in [-1, 1]
        img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor * 2.0 - 1.0).to(target_device, dtype=self.dtype)

        # Move VAE to GPU explicitly only for model offload mode.
        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        # Encode to latent
        with torch.no_grad():
            latent = self.pipe.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * self.pipe.vae.config.scaling_factor

        return latent

    def encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Batch encode images [B,C,H,W] in [-1,1] to latents [B,C,H/8,W/8]."""
        target_device = self._get_execution_device()
        images = images.to(target_device, dtype=self.dtype)

        # Move VAE to GPU explicitly only for model offload mode.
        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            latents = self.pipe.vae.encode(images).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor

        return latents

    def decode_latent_to_image(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to PIL image using the VAE."""
        import numpy as np

        target_device = self._get_execution_device()
        latents = latents.to(target_device, dtype=self.dtype)
        latents = latents / self.pipe.vae.config.scaling_factor

        # Move VAE to GPU explicitly only for model offload mode.
        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.float().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """Batch decode latents [B,C,H/8,W/8] to images [B,C,H,W] in [0,1]."""
        target_device = self._get_execution_device()
        latents = latents.to(target_device, dtype=self.dtype)
        latents = latents / self.pipe.vae.config.scaling_factor

        # Move VAE to GPU explicitly only for model offload mode.
        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            images = self.pipe.vae.decode(latents).sample

        # Convert from [-1,1] to [0,1]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images
