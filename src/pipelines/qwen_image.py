"""
Qwen Image pipeline implementation.
"""

import gc
import logging
import math
import os
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)

# Environment variable to enable aggressive memory cleanup (default: off for faster iterative runs)
AITK_AGGRESSIVE_CLEANUP = os.environ.get("AITK_AGGRESSIVE_CLEANUP", "").lower() in ("1", "true", "yes")

# Lazy imports - diffusers is only imported when actually needed
# This avoids loading all diffusers pipelines at startup
# CONDITION_IMAGE_SIZE is fetched lazily in _get_condition_image_size()
_CONDITION_IMAGE_SIZE_CACHE = None


def _get_condition_image_size():
    """Get CONDITION_IMAGE_SIZE lazily to avoid importing diffusers at module load."""
    global _CONDITION_IMAGE_SIZE_CACHE
    if _CONDITION_IMAGE_SIZE_CACHE is None:
        try:
            from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
                CONDITION_IMAGE_SIZE,
            )
            _CONDITION_IMAGE_SIZE_CACHE = CONDITION_IMAGE_SIZE
        except ImportError:
            # Fallback value if diffusers version doesn't have this
            _CONDITION_IMAGE_SIZE_CACHE = 147456  # ~384x384
    return _CONDITION_IMAGE_SIZE_CACHE



class QwenImagePipeline(BasePipeline):
    """Qwen Image (DiT / flow-matching) pipeline wrapper.

    Key detail for latent workflows:
    - Diffusers returns *packed* latents for Qwen Image when `output_type='latent'`.
      Shape is typically [B, N, C*4] where N = (H/16)*(W/16) and C is the VAE latent channels.
    - For ComfyUI-style latent workflows (LatentUpscale + second pass), we expose *spatial* VAE latents
      in the standard ComfyUI tensor shape [B, C, H/8, W/8].

    We convert between packed <-> spatial using the helper methods implemented in the diffusers
    Qwen pipelines (`_pack_latents` / `_unpack_latents`).
    """
    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE,
        base_model="Qwen/Qwen-Image",
        default_steps=25,
        default_guidance_scale=4.0,
        # Qwen Image pipelines require height/width divisible by (vae_scale_factor * 2).
        # With vae_scale_factor=8, that is 16.
        resolution_divisor=16,
        supports_negative_prompt=True,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._img2img_pipe = None

    def _enable_sequential_cpu_offload(self):
        """Enable sequential CPU offload for lower RAM usage (slower inference).
        
        Overrides BasePipeline._enable_sequential_cpu_offload() to sync 
        the img2img pipeline after enabling offload on the base pipeline.
        """
        # Call base implementation
        super()._enable_sequential_cpu_offload()
        # Sync img2img pipe if already created
        self._sync_img2img_execution_device()

    def _load_pipeline(self):
        """Load the base Qwen pipeline. img2img is created in load() after device setup."""
        from diffusers import QwenImagePipeline as DiffusersQwenImagePipeline

        self.pipe = DiffusersQwenImagePipeline.from_pretrained(
            self.CONFIG.base_model,
            torch_dtype=self.dtype,
            token=self.hf_token,
            local_files_only=False,
        )
        # NOTE: _img2img_pipe is created in load() after CPU offload is enabled,
        # so it inherits the proper execution-device state.

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Load pipeline, then create img2img wrapper with proper device state."""
        super().load(lora_paths, lora_scale)
        self._create_img2img_pipeline()

    def _create_img2img_pipeline(self):
        """Create img2img pipeline wrapper sharing components with base pipeline."""
        try:
            from diffusers import QwenImageImg2ImgPipeline

            # Share weights/components to avoid duplicate VRAM usage.
            # Created AFTER base load() so CPU offload hooks are already in place.
            self._img2img_pipe = QwenImageImg2ImgPipeline(**self.pipe.components)
            # Sync execution device state
            self._sync_img2img_execution_device()
            logger.debug("QwenImageImg2ImgPipeline created for latent refinement")
        except Exception as e:
            logger.warning(
                f"QwenImageImg2ImgPipeline not available in your diffusers install: {e}. "
                "Latent refinement will be unavailable (update diffusers)."
            )
            self._img2img_pipe = None

    def _sync_img2img_execution_device(self):
        """Sync execution device/offload state from base pipe to img2img pipe.
        
        This ensures the img2img pipeline inherits the same offload hooks and
        device placement as the base pipeline.
        """
        if self._img2img_pipe is None or self.pipe is None:
            return

        # Copy execution device if set (diffusers pipelines store this for input handling)
        if hasattr(self.pipe, "_execution_device"):
            try:
                # Some diffusers versions use a property, others use an attribute
                exec_device = self.pipe._execution_device
                if hasattr(self._img2img_pipe, "_execution_device"):
                    # Try to set it (may be read-only property in some versions)
                    pass  # Usually synced via shared components
            except Exception:
                pass

        # For sequential offload, the hooks are on the shared components,
        # so they should work automatically. For model_cpu_offload, same applies.
        # The key is that we create _img2img_pipe AFTER offload is enabled.
        logger.debug("img2img pipeline execution device synced")

    def unload(self):
        """Unload pipeline and clear img2img wrapper to free VRAM."""
        # Clear img2img pipeline first (it holds references to shared components)
        if getattr(self, "_img2img_pipe", None) is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
            logger.debug("Cleared _img2img_pipe")

        # Call base unload (clears self.pipe, gc.collect, empty_cache)
        super().unload()

    # ===== Latent helpers =====

    def _vae_scale_factor(self) -> int:
        return int(getattr(self.pipe, "vae_scale_factor", 8))

    def _latent_stats(self, device: torch.device, dtype: torch.dtype):
        """Return (mean, std) tensors for Qwen VAE latent normalization, or (None, None)."""
        cfg = getattr(self.pipe, "vae", None)
        cfg = getattr(cfg, "config", None)
        if cfg is None:
            return None, None
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if mean is None or std is None:
            return None, None
        mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        return mean_t, std_t

    def _packed_to_spatial(self, packed: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Convert diffusers packed Qwen latents [B,N,D] -> spatial latents [B,C,H/8,W/8]."""
        if packed.ndim != 3:
            raise ValueError(f"Expected packed latents [B,N,D], got shape {tuple(packed.shape)}")
        if not hasattr(self.pipe, "_unpack_latents"):
            raise NotImplementedError(
                "Your diffusers QwenImagePipeline is missing _unpack_latents; update diffusers."
            )
        lat5d = self.pipe._unpack_latents(packed, height, width, self._vae_scale_factor())
        if lat5d.ndim == 5:
            return lat5d[:, :, 0]  # [B,C,H_lat,W_lat]
        if lat5d.ndim == 4:
            return lat5d
        raise ValueError(f"Unexpected unpacked latent shape: {tuple(lat5d.shape)}")

    def _spatial_to_packed(self, latents: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Convert spatial latents [B,C,H/8,W/8] -> packed latents [B,N,D]."""
        if latents.ndim == 4:
            lat5d = latents.unsqueeze(2)  # [B,C,1,H_lat,W_lat]
        elif latents.ndim == 5:
            lat5d = latents
        else:
            raise ValueError(f"Expected spatial latents [B,C,H,W] or [B,C,1,H,W], got {tuple(latents.shape)}")

        if not hasattr(self.pipe, "_pack_latents"):
            raise NotImplementedError(
                "Your diffusers QwenImagePipeline is missing _pack_latents; update diffusers."
            )

        # Qwen pack/unpack expects [B,1,C,H,W] ordering.
        lat5d = lat5d.transpose(1, 2)  # [B,1,C,H,W]
        b, _, c, h, w = lat5d.shape
        return self.pipe._pack_latents(lat5d, b, c, h, w)

    def _ensure_target_latent_size(self, latents_4d: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Upscale/downscale spatial latents to match the target output resolution."""
        vae_sf = self._vae_scale_factor()
        target_h = 2 * (int(height) // (vae_sf * 2))
        target_w = 2 * (int(width) // (vae_sf * 2))
        if latents_4d.shape[-2:] == (target_h, target_w):
            return latents_4d
        # Bilinear is usually the most stable choice for latent interpolation.
        return F.interpolate(latents_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)

    # ===== Encode / decode =====

    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to *spatial* standardized latents [B,C,H/8,W/8]."""
        import numpy as np

        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        target_device = self._get_execution_device()

        img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor * 2.0 - 1.0).to(target_device, dtype=self.dtype)
        img_tensor = img_tensor.unsqueeze(2)  # [B,3,1,H,W]

        # With sequential_cpu_offload, VAE uses meta tensors with hooks - don't call .to()
        # With model_cpu_offload, we need to move VAE to device explicitly
        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            lat = self.pipe.vae.encode(img_tensor).latent_dist.sample()

        mean, std = self._latent_stats(device=lat.device, dtype=lat.dtype)
        if mean is not None and std is not None:
            lat = (lat - mean) / std

        # Drop the temporal dim (always 1 for images).
        if lat.ndim == 5:
            lat = lat[:, :, 0]
        return lat

    def encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Batch encode images [B,C,H,W] in [-1,1] to spatial latents [B,C,H/8,W/8].
        
        This is more efficient than looping encode_image_to_latent for batches.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        target_device = self._get_execution_device()
        images = images.to(target_device, dtype=self.dtype)
        
        # Qwen VAE expects 5D input [B,C,1,H,W]
        images_5d = images.unsqueeze(2)

        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            latents = self.pipe.vae.encode(images_5d).latent_dist.sample()

        mean, std = self._latent_stats(device=latents.device, dtype=latents.dtype)
        if mean is not None and std is not None:
            latents = (latents - mean) / std

        # Drop temporal dim -> [B,C,H,W]
        if latents.ndim == 5:
            latents = latents[:, :, 0]
        return latents

    def decode_latent_to_image(self, latents: torch.Tensor) -> Image.Image:
        """Decode *spatial* standardized latents [B,C,H/8,W/8] to a PIL image."""
        import numpy as np

        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        target_device = self._get_execution_device()

        if latents.ndim == 3:
            # If someone passes packed latents, try to infer a square size.
            n = latents.shape[1]
            side = int(math.sqrt(n))
            if side * side != n:
                raise ValueError(
                    "Packed Qwen latents are ambiguous without height/width. "
                    "Pass spatial latents [B,C,H,W] instead."
                )
            # Infer an image size assuming square.
            vae_sf = self._vae_scale_factor()
            img_h = img_w = (side * 2) * vae_sf
            latents = self._packed_to_spatial(latents, height=img_h, width=img_w)

        if latents.ndim == 4:
            lat5d = latents.to(target_device, dtype=self.dtype).unsqueeze(2)
        elif latents.ndim == 5:
            lat5d = latents.to(target_device, dtype=self.dtype)
        else:
            raise ValueError(f"Unexpected latents shape: {tuple(latents.shape)}")

        mean, std = self._latent_stats(device=lat5d.device, dtype=lat5d.dtype)
        if mean is not None and std is not None:
            lat5d = lat5d * std + mean

        # With sequential_cpu_offload, VAE uses meta tensors with hooks - don't call .to()
        # With model_cpu_offload, we need to move VAE to device explicitly
        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            img = self.pipe.vae.decode(lat5d).sample

        # Drop temporal dim if present.
        if img.ndim == 5:
            img = img[:, :, 0]

        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.float().cpu().permute(0, 2, 3, 1).numpy()[0]
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """Batch decode spatial latents [B,C,H/8,W/8] to images [B,C,H,W] in [0,1].
        
        This is more efficient than looping decode_latent_to_image for batches.
        Returns tensor in [0,1] range with shape [B,C,H,W].
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        target_device = self._get_execution_device()

        # Handle spatial latents [B,C,H,W] -> 5D [B,C,1,H,W]
        if latents.ndim == 4:
            lat5d = latents.to(target_device, dtype=self.dtype).unsqueeze(2)
        elif latents.ndim == 5:
            lat5d = latents.to(target_device, dtype=self.dtype)
        else:
            raise ValueError(f"Expected spatial latents [B,C,H,W], got {tuple(latents.shape)}")

        mean, std = self._latent_stats(device=lat5d.device, dtype=lat5d.dtype)
        if mean is not None and std is not None:
            lat5d = lat5d * std + mean

        if self.offload_mode == "model":
            self.pipe.vae.to(target_device)

        with torch.no_grad():
            images = self.pipe.vae.decode(lat5d).sample

        # Drop temporal dim if present
        if images.ndim == 5:
            images = images[:, :, 0]

        # Convert from [-1,1] to [0,1]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    # ===== Inference =====

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
        fps: int = 24,
        output_type: str = "pil",
        latents: Optional[torch.Tensor] = None,
        denoise_strength: float = 1.0,
    ) -> Dict[str, Any]:

        if control_image is not None or control_images is not None:
            logger.warning("Qwen Image base pipeline does not support control images; ignoring")

        diffusers_out = "latent" if output_type == "latent" else "pil"

        if latents is None:
            # Text-to-image
            call_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": guidance_scale,
                "generator": generator,
                "output_type": diffusers_out,
            }
            # Comfy-native progress + interrupt (no-op unless an observer is installed).
            self._inject_diffusers_callback_kwargs(call_kwargs, total_steps=num_inference_steps, pipe=self.pipe)
            result = self.pipe(**call_kwargs)

            if output_type == "latent":
                out = getattr(result, "images", None)
                if out is None:
                    raise ValueError("Diffusers Qwen pipeline returned no latents")
                # Convert packed -> spatial for ComfyUI compatibility.
                if isinstance(out, torch.Tensor) and out.ndim == 3:
                    out = self._packed_to_spatial(out, height=height, width=width)
                # Optionally clear GPU cache (off by default for faster iterative runs)
                if AITK_AGGRESSIVE_CLEANUP and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"latents": out}

            # PIL
            return {"image": result.images[0]}

        # Latent refinement (img2img)
        if self._img2img_pipe is None:
            raise NotImplementedError(
                "Qwen latent refinement requires QwenImageImg2ImgPipeline (update diffusers)."
            )

        # Optionally clear GPU cache before refine step (off by default)
        if AITK_AGGRESSIVE_CLEANUP and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Accept either spatial latents [B,C,H,W] or packed [B,N,D].
        if latents.ndim == 3:
            lat4d = self._packed_to_spatial(latents, height=height, width=width)
        elif latents.ndim == 5:
            lat4d = latents[:, :, 0]
        elif latents.ndim == 4:
            lat4d = latents
        else:
            raise ValueError(f"Unsupported latents shape: {tuple(latents.shape)}")

        lat4d = self._ensure_target_latent_size(lat4d, height=height, width=width)

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": lat4d,
            "strength": float(denoise_strength),
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": guidance_scale,
            "generator": generator,
            "output_type": diffusers_out,
        }
        # Comfy-native progress + interrupt (no-op unless an observer is installed).
        self._inject_diffusers_callback_kwargs(call_kwargs, total_steps=num_inference_steps, pipe=self._img2img_pipe)
        result = self._img2img_pipe(**call_kwargs)

        # Optionally free memory after img2img inference (off by default)
        del lat4d
        if AITK_AGGRESSIVE_CLEANUP:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if output_type == "latent":
            out = getattr(result, "images", None)
            if out is None:
                raise ValueError("Diffusers Qwen img2img pipeline returned no latents")
            if isinstance(out, torch.Tensor) and out.ndim == 3:
                out = self._packed_to_spatial(out, height=height, width=width)
            elif isinstance(out, torch.Tensor) and out.ndim == 5:
                out = out[:, :, 0]
            return {"latents": out}

        return {"image": result.images[0]}


class QwenImage2512Pipeline(QwenImagePipeline):
    """Qwen Image 2512 version."""

    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE_2512,
        base_model="Qwen/Qwen-Image-2512",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        lora_merge_method=LoraMergeMethod.FUSE,
    )


class QwenImageEditPipeline(BasePipeline):
    """
    Qwen Image Edit pipeline for image editing.

    Key: encode_prompt requires control_images (aligned with ai-toolkit).
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE_EDIT,
        base_model="Qwen/Qwen-Image-Edit",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
    )

    def _load_pipeline(self):
        """Load Qwen Image Edit pipeline."""
        try:
            from diffusers import QwenImageEditPipeline as DiffusersQwenImageEditPipeline

            self.pipe = DiffusersQwenImageEditPipeline.from_pretrained(
                self.CONFIG.base_model,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        except ImportError:
            logger.warning("QwenImagePipeline not available, using placeholder")
            raise NotImplementedError("QwenImagePipeline not available in this diffusers version")

    def _preprocess_ctrl_for_encode(self, ctrl_pil: Image.Image, device: str, dtype) -> torch.Tensor:
        """
        Preprocess control image for prompt encoding (aligned with ai-toolkit).
        Resize to ~1MP while keeping aspect ratio.
        """
        ctrl_tensor = TF.to_tensor(ctrl_pil).unsqueeze(0).to(device, dtype=dtype)

        # images are always run through at 1MP, based on ai-toolkit qwen_image_edit.py
        target_area = 1024 * 1024
        ratio = ctrl_tensor.shape[2] / ctrl_tensor.shape[3]
        w = math.sqrt(target_area * ratio)
        h = w / ratio
        w = round(w / 32) * 32
        h = round(h / 32) * 32

        ctrl_tensor = F.interpolate(ctrl_tensor, size=(int(h), int(w)), mode="bilinear")
        return ctrl_tensor

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
        """Run Qwen Image Edit inference with control_images in encode_prompt (aligned with ai-toolkit)."""
        if control_image is None:
            raise ValueError("Qwen Image Edit requires a control image")

        device = self._get_execution_device()
        dtype = self.dtype

        # Preprocess control image for prompt encoding (aligned with ai-toolkit)
        ctrl_tensor = self._preprocess_ctrl_for_encode(control_image, device, dtype)

        # Encode prompt with control image (aligned with ai-toolkit qwen_image_edit.py)
        prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
            prompt,
            image=ctrl_tensor,
            device=device,
            num_images_per_prompt=1,
        )

        # Encode negative prompt with control image
        neg_prompt_embeds, neg_prompt_embeds_mask = self.pipe.encode_prompt(
            negative_prompt or "",
            image=ctrl_tensor,
            device=device,
            num_images_per_prompt=1,
        )

        # Resize control image for generation
        control_image = control_image.resize((width, height), Image.LANCZOS)

        call_kwargs = {
            "image": control_image,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": neg_prompt_embeds,
            "negative_prompt_embeds_mask": neg_prompt_embeds_mask,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": guidance_scale,
            "generator": generator,
        }
        # Comfy-native progress + interrupt (no-op unless an observer is installed).
        self._inject_diffusers_callback_kwargs(call_kwargs, total_steps=num_inference_steps, pipe=self.pipe)
        result = self.pipe(**call_kwargs)

        return {"image": result.images[0]}


class QwenImageEditPlusPipeline(BasePipeline):
    """
    Qwen Image Edit Plus pipeline for multi-image editing.

    Supports up to 3 control images.
    Key: Uses tensor_encode for prompt encoding (aligned with ai-toolkit).
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE_EDIT_PLUS_2509,
        base_model="Qwen/Qwen-Image-Edit-2509",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
    )

    def _load_pipeline(self):
        """Load Qwen Image Edit Plus pipeline."""
        try:
            from diffusers import QwenImageEditPlusPipeline as DiffusersQwenImageEditPlusPipeline

            self.pipe = DiffusersQwenImageEditPlusPipeline.from_pretrained(
                self.CONFIG.base_model,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        except ImportError:
            logger.warning("QwenImageEditPlusPipeline not available, using placeholder")
            raise NotImplementedError("QwenImageEditPlusPipeline not available in this diffusers version")

    def _preprocess_ctrl_for_encode(self, ctrl_pil: Image.Image, device: str, dtype) -> torch.Tensor:
        """
        Preprocess control image for prompt encoding (aligned with ai-toolkit).
        1. Convert to 0-1 tensor
        2. Resize to CONDITION_IMAGE_SIZE
        """
        ctrl_tensor = TF.to_tensor(ctrl_pil).unsqueeze(0).to(device, dtype=dtype)

        # Calculate target size based on CONDITION_IMAGE_SIZE (keep aspect ratio)
        ratio = ctrl_tensor.shape[2] / ctrl_tensor.shape[3]
        w = math.sqrt(_get_condition_image_size() * ratio)
        h = w / ratio
        w = round(w / 32) * 32
        h = round(h / 32) * 32

        ctrl_tensor = F.interpolate(ctrl_tensor, size=(int(h), int(w)), mode="bilinear")
        return ctrl_tensor

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
        """Run Qwen Image Edit Plus inference with tensor_encode (aligned with ai-toolkit)."""
        # Collect control images
        images = []
        if control_images:
            images = list(control_images)
        elif control_image:
            images = [control_image]

        if not images:
            raise ValueError("Qwen Image Edit Plus requires at least one control image")

        device = self._get_execution_device()
        dtype = self.dtype

        # Preprocess control images for prompt encoding
        ctrl_tensors = [self._preprocess_ctrl_for_encode(img, device, dtype) for img in images]

        # Encode prompt with tensor images (aligned with ai-toolkit)
        prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
            prompt,
            image=ctrl_tensors,
            device=device,
            num_images_per_prompt=1,
        )

        # Encode negative prompt
        neg_prompt_embeds, neg_prompt_embeds_mask = self.pipe.encode_prompt(
            negative_prompt or "",
            image=ctrl_tensors,
            device=device,
            num_images_per_prompt=1,
        )

        # Run inference with pre-encoded prompts
        call_kwargs = {
            "image": images,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": neg_prompt_embeds,
            "negative_prompt_embeds_mask": neg_prompt_embeds_mask,
            "generator": generator,
            "true_cfg_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
        }
        # Comfy-native progress + interrupt (no-op unless an observer is installed).
        self._inject_diffusers_callback_kwargs(call_kwargs, total_steps=num_inference_steps, pipe=self.pipe)
        result = self.pipe(**call_kwargs)

        return {"image": result.images[0]}


class QwenImageEditPlus2509Pipeline(QwenImageEditPlusPipeline):
    """Qwen Image Edit Plus 2509 version."""

    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE_EDIT_PLUS_2509,
        base_model="Qwen/Qwen-Image-Edit-2509",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
    )


class QwenImageEditPlus2511Pipeline(QwenImageEditPlusPipeline):
    """Qwen Image Edit Plus 2511 version."""

    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE_EDIT_PLUS_2511,
        base_model="Qwen/Qwen-Image-Edit-2511",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
        lora_merge_method=LoraMergeMethod.FUSE,
    )
