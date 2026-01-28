"""
Qwen Image pipeline implementation.
"""

import logging
import math
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)

# Import constants from diffusers official pipeline
try:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
    )
except ImportError:
    # Fallback value if diffusers version doesn't have this
    CONDITION_IMAGE_SIZE = 147456  # ~384x384


class QwenImagePipeline(BasePipeline):
    """
    Qwen Image pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE,
        base_model="Qwen/Qwen-Image",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
    )

    def _load_pipeline(self):
        """Load Qwen Image pipeline."""
        # Import here to avoid circular imports
        try:
            from diffusers import QwenImagePipeline as DiffusersQwenImagePipeline

            self.pipe = DiffusersQwenImagePipeline.from_pretrained(
                self.CONFIG.base_model,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        except ImportError:
            # Fallback if QwenImagePipeline not available
            logger.warning("QwenImagePipeline not available, using placeholder")
            raise NotImplementedError("QwenImagePipeline not available in this diffusers version")

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
        """Run Qwen Image inference."""
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale,
            generator=generator,
            negative_prompt=negative_prompt,
        )

        return {"image": result.images[0]}


class QwenImage2512Pipeline(QwenImagePipeline):
    """Qwen Image 2512 version."""

    CONFIG = PipelineConfig(
        model_type=ModelType.QWEN_IMAGE_2512,
        base_model="Qwen/Qwen-Image-2512",
        resolution_divisor=32,
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

        device = "cuda"
        dtype = torch.bfloat16

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

        result = self.pipe(
            image=control_image,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_prompt_embeds_mask=neg_prompt_embeds_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale,
            generator=generator,
        )

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
        w = math.sqrt(CONDITION_IMAGE_SIZE * ratio)
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

        device = "cuda"
        dtype = torch.bfloat16

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
        result = self.pipe(
            image=images,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_prompt_embeds_mask=neg_prompt_embeds_mask,
            generator=generator,
            true_cfg_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        )

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
