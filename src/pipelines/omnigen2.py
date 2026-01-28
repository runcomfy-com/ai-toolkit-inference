"""
OmniGen2 pipeline implementation.

Uses custom ai-toolkit OmniGen2Pipeline with Qwen2.5-VL text encoder.
Supports text-to-image and image editing with reference images.
"""

import gc
import logging
import os
import sys
from typing import Dict, Any, Optional, List

import torch
from PIL import Image
from safetensors.torch import load_file

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType
from ..config import settings

logger = logging.getLogger(__name__)

# Add ai-toolkit path for imports (configurable via AI_TOOLKIT_PATH env var)
if os.path.exists(settings.ai_toolkit_path) and settings.ai_toolkit_path not in sys.path:
    sys.path.insert(0, settings.ai_toolkit_path)


class OmniGen2Pipeline(BasePipeline):
    """
    OmniGen2 multi-modal image generation pipeline.

    Uses ai-toolkit's custom OmniGen2Pipeline with:
    - Qwen2.5-VL for text/vision encoding
    - OmniGen2Transformer2DModel for image generation
    - FLUX-style VAE (16 latent channels)
    - FlowMatchEulerDiscreteScheduler with dynamic_time_shift

    Key features:
    - Text-to-image generation
    - Image editing with reference images (up to 5)
    - Dual guidance (text + image CFG)

    LoRA format:
    - ai-toolkit saves with "diffusion_model." prefix (ComfyUI format)
    - diffusers expects "transformer." prefix
    - Conversion is handled in _load_lora()
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.OMNIGEN2,
        base_model="OmniGen2/OmniGen2",
        resolution_divisor=16,  # vae_scale_factor (8) * patch_size (2)
        default_steps=25,  # Match training sample config
        default_guidance_scale=4.0,
        requires_control_image=False,  # Optional but supported
        supports_negative_prompt=True,
        lora_merge_method=LoraMergeMethod.CUSTOM,  # Manual LoRA merge
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mllm = None
        self.processor = None
        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.freqs_cis = None

    def _load_pipeline(self):
        """Load OmniGen2 pipeline using ai-toolkit components."""
        try:
            # Import custom components from ai-toolkit
            from extensions_built_in.diffusion_models.omnigen2.src.models.transformers import OmniGen2Transformer2DModel
            from extensions_built_in.diffusion_models.omnigen2.src.models.transformers.repo import (
                OmniGen2RotaryPosEmbed,
            )
            from extensions_built_in.diffusion_models.omnigen2.src.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler as OmniGen2Scheduler,
            )
            from extensions_built_in.diffusion_models.omnigen2.src.pipelines.omnigen2.pipeline_omnigen2 import (
                OmniGen2Pipeline as AitkOmniGen2Pipeline,
            )

            # Standard imports
            from diffusers import AutoencoderKL
            from transformers import Qwen2_5_VLForConditionalGeneration, CLIPProcessor

            model_path = self.CONFIG.base_model

            # ========== Step 1: Load MLLM (Qwen2.5-VL) ==========
            logger.info("[1/4] Loading Qwen2.5-VL MLLM")
            processor = CLIPProcessor.from_pretrained(
                model_path,
                subfolder="processor",
                use_fast=True,
                token=self.hf_token,
            )

            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                subfolder="mllm",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            mllm.to(self.device, dtype=self.dtype)
            mllm.eval()
            mllm.requires_grad_(False)

            # ========== Step 2: Load Transformer ==========
            logger.info("[2/4] Loading OmniGen2 transformer")
            transformer = OmniGen2Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            transformer.to(self.device, dtype=self.dtype)
            transformer.eval()
            transformer.requires_grad_(False)

            # ========== Step 3: Load VAE ==========
            logger.info("[3/4] Loading VAE")
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            vae.to(self.device, dtype=self.dtype)
            vae.eval()
            vae.requires_grad_(False)

            # ========== Step 4: Create Scheduler and Pipeline ==========
            logger.info("[4/4] Creating scheduler and pipeline")
            scheduler = OmniGen2Scheduler(
                num_train_timesteps=1000,
                dynamic_time_shift=True,
            )

            # Create the custom pipeline
            self.pipe = AitkOmniGen2Pipeline(
                transformer=transformer,
                vae=vae,
                scheduler=scheduler,
                mllm=mllm,
                processor=processor,
            )

            # Store component references
            self.mllm = mllm
            self.processor = processor
            self.transformer = transformer
            self.vae = vae
            self.scheduler = scheduler

            # Pre-compute rotary position embeddings
            self.freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
                transformer.config.axes_dim_rope,
                transformer.config.axes_lens,
                theta=10000,
            )

            # Print GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

            logger.info("OmniGen2 pipeline loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import OmniGen2 components: {e}")
            logger.error(f"Make sure AI_TOOLKIT_PATH ({settings.ai_toolkit_path}) is valid")
            raise
        except Exception as e:
            logger.error(f"Failed to load OmniGen2 pipeline: {e}")
            raise

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """
        Load LoRA weights for OmniGen2 using fused mode.

        Since the custom OmniGen2Pipeline doesn't inherit LoraLoaderMixin,
        we fuse LoRA weights directly into the transformer.

        Note: Dynamic scale changes are NOT supported after loading.

        ai-toolkit saves LoRA with "diffusion_model." prefix (ComfyUI format).
        We strip this prefix and fuse directly into transformer weights.

        Args:
            lora_paths: List of LoRA file paths (uses first one)
            lora_scale: LoRA weight scale (applied at load time, cannot be changed later)
        """
        if not lora_paths:
            logger.warning("No LoRA paths provided")
            return

        lora_path = lora_paths[0]
        if isinstance(lora_path, dict):
            lora_path = list(lora_path.values())[0] if lora_path else None

        if not lora_path or not os.path.exists(lora_path):
            logger.warning(f"LoRA file not found: {lora_path}")
            return

        logger.info(f"Loading LoRA (fused mode): {lora_path}")

        # 1. Load raw LoRA (ComfyUI format: diffusion_model.xxx)
        lora_sd = load_file(lora_path)

        # 2. Convert keys: remove "diffusion_model." prefix
        converted_sd = {}
        for k, v in lora_sd.items():
            new_key = k.replace("diffusion_model.", "")
            converted_sd[new_key] = v

        # 3. Group LoRA A and B weights
        lora_pairs = {}
        for k, v in converted_sd.items():
            if "lora_A" in k:
                base_key = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["A"] = v
            elif "lora_B" in k:
                base_key = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["B"] = v

        # 4. Fuse LoRA weights into model: W' = W + scale * B @ A
        model_sd = self.transformer.state_dict()
        updated = 0
        for base_key, pair in lora_pairs.items():
            if "A" in pair and "B" in pair:
                weight_key = f"{base_key}.weight"
                if weight_key in model_sd:
                    delta = lora_scale * (pair["B"] @ pair["A"])
                    model_sd[weight_key] = model_sd[weight_key] + delta.to(
                        model_sd[weight_key].device, dtype=model_sd[weight_key].dtype
                    )
                    updated += 1

        if updated > 0:
            self.transformer.load_state_dict(model_sd)
            self.lora_loaded = True
            self._current_lora_scale = lora_scale
            logger.info(f"Fused {updated} LoRA weight pairs with scale={lora_scale}")
        else:
            logger.warning("No LoRA weights were fused")

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
    ) -> Dict[str, Any]:
        """Run OmniGen2 inference."""
        # Prepare input images (reference images for editing)
        input_images = []
        if control_image is not None:
            input_images = [control_image]
        if control_images is not None and len(control_images) > 0:
            input_images = control_images

        # If no images, pass None
        if len(input_images) == 0:
            input_images = None

        # CRITICAL: Match training behavior where chat template is applied twice
        # Training: get_prompt_embeds applies template, then pipeline.encode_prompt applies again
        # We apply once here, so pipeline.encode_prompt applies the second time
        prompt = self.pipe._apply_chat_template(prompt)
        neg = negative_prompt if negative_prompt else ""
        if neg:
            neg = self.pipe._apply_chat_template(neg)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=neg,
            input_images=input_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=guidance_scale,
            image_guidance_scale=1.0,  # Default, can be exposed later
            generator=generator,
            align_res=False,  # Use specified resolution
            max_sequence_length=256,  # Match training (explicit 256)
            return_dict=True,
        )

        return {"image": result.images[0]}

    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        if self.mllm is not None:
            del self.mllm
            self.mllm = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.vae is not None:
            del self.vae
            self.vae = None
        if self.scheduler is not None:
            del self.scheduler
            self.scheduler = None
        if self.freqs_cis is not None:
            del self.freqs_cis
            self.freqs_cis = None

        self.lora_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("OmniGen2 pipeline unloaded")
