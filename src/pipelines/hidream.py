"""
HiDream pipeline implementation.

Uses ai-toolkit's custom HiDreamImagePipeline with:
- 4 text encoders: CLIP×2, T5, Llama-3.1-8B
- HiDreamImageTransformer2DModel (sparse DiT with MoE)
- FlowUniPCMultistepScheduler
- FLUX-style VAE (16 latent channels)
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

# Default Llama model path (same as ai-toolkit)
LLAMA_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct"


class HiDreamPipeline(BasePipeline):
    """
    HiDream text-to-image pipeline.

    Uses ai-toolkit's HiDreamImagePipeline with:
    - 4 text encoders: CLIP×2 (pooled), T5 (sequence), Llama-3.1-8B (hidden states)
    - FlowUniPCMultistepScheduler (shift=3.0)

    LoRA format:
    - ai-toolkit saves with "diffusion_model." prefix (ComfyUI format)
    - Fused into transformer weights at load time
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.HIDREAM,
        base_model="HiDream-ai/HiDream-I1-Full",
        resolution_divisor=16,
        default_steps=50,
        default_guidance_scale=5.0,
        lora_merge_method=LoraMergeMethod.CUSTOM,  # Manual LoRA merge
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.vae = None

    def _load_pipeline(self):
        """Load HiDream pipeline using ai-toolkit components."""
        # Import ai-toolkit components
        from extensions_built_in.diffusion_models.hidream.src.models.transformers.transformer_hidream_image import (
            HiDreamImageTransformer2DModel,
        )
        from extensions_built_in.diffusion_models.hidream.src.schedulers.fm_solvers_unipc import (
            FlowUniPCMultistepScheduler,
        )
        from extensions_built_in.diffusion_models.hidream.src.pipelines.hidream_image.pipeline_hidream_image import (
            HiDreamImagePipeline as AitkPipeline,
        )
        from diffusers import AutoencoderKL
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5Tokenizer,
            LlamaForCausalLM,
            PreTrainedTokenizerFast,
        )

        model_path = self.CONFIG.base_model

        # Step 1: Load Llama 8B
        logger.info("[1/4] Loading Llama 8B")
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_PATH, use_fast=False, token=self.hf_token)
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_PATH,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=self.dtype,
            token=self.hf_token,
        ).to(self.device)
        text_encoder_4.eval().requires_grad_(False)

        # Step 2: Load transformer
        logger.info("[2/4] Loading transformer")
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
            token=self.hf_token,
        ).to(self.device)
        transformer.eval().requires_grad_(False)

        # Step 3: Load VAE + CLIP + T5
        logger.info("[3/4] Loading VAE and text encoders")
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        vae.eval().requires_grad_(False)

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", token=self.hf_token)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", token=self.hf_token)
        text_encoder_3 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_3", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        tokenizer_3 = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer_3", token=self.hf_token)

        for te in [text_encoder, text_encoder_2, text_encoder_3]:
            te.eval().requires_grad_(False)

        # Step 4: Create pipeline
        logger.info("[4/4] Creating pipeline")
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)

        self.pipe = AitkPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
            text_encoder_4=text_encoder_4,
            tokenizer_4=tokenizer_4,
            transformer=transformer,
        )
        self.transformer = transformer
        self.vae = vae

        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info("HiDream pipeline loaded")

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """Load LoRA using fused mode (same as OmniGen2)."""
        if not lora_paths:
            return

        lora_path = lora_paths[0]
        if isinstance(lora_path, dict):
            lora_path = list(lora_path.values())[0] if lora_path else None
        if not lora_path or not os.path.exists(lora_path):
            logger.warning(f"LoRA not found: {lora_path}")
            return

        logger.info(f"Loading LoRA (fused): {lora_path}")
        lora_sd = load_file(lora_path)

        # Convert: diffusion_model.xxx -> xxx
        converted = {k.replace("diffusion_model.", ""): v for k, v in lora_sd.items()}

        # Group LoRA pairs
        pairs = {}
        for k, v in converted.items():
            if "lora_A" in k:
                base = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                pairs.setdefault(base, {})["A"] = v
            elif "lora_B" in k:
                base = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                pairs.setdefault(base, {})["B"] = v

        # Fuse: W' = W + scale * B @ A
        model_sd = self.transformer.state_dict()
        updated = 0
        for base, pair in pairs.items():
            if "A" in pair and "B" in pair:
                key = f"{base}.weight"
                if key in model_sd:
                    delta = lora_scale * (pair["B"] @ pair["A"])
                    model_sd[key] = model_sd[key] + delta.to(model_sd[key].device, dtype=model_sd[key].dtype)
                    updated += 1

        if updated:
            self.transformer.load_state_dict(model_sd)
            self.lora_loaded = True
            self._current_lora_scale = lora_scale
            logger.info(f"Fused {updated} LoRA pairs, scale={lora_scale}")

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
        with torch.cuda.amp.autocast(dtype=self.dtype):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=128,
            )
        return {"image": result.images[0]}

    def unload(self):
        for attr in ["pipe", "transformer", "vae"]:
            if getattr(self, attr, None):
                delattr(self, attr)
        self.lora_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HiDream unloaded")


class HiDreamE1Pipeline(BasePipeline):
    """
    HiDream E1 image editing pipeline.

    Uses HiDreamImageEditingPipeline with image_guidance_scale support.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.HIDREAM_E1,
        base_model="HiDream-ai/HiDream-E1-Full",
        resolution_divisor=16,
        default_steps=28,
        default_guidance_scale=5.0,
        requires_control_image=True,
        lora_merge_method=LoraMergeMethod.CUSTOM,  # Manual LoRA merge
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.vae = None

    def _load_pipeline(self):
        """Load HiDream-E1 pipeline."""
        from extensions_built_in.diffusion_models.hidream.src.schedulers.fm_solvers_unipc import (
            FlowUniPCMultistepScheduler,
        )
        from extensions_built_in.diffusion_models.hidream.src.pipelines.hidream_image.pipeline_hidream_image_editing import (
            HiDreamImageEditingPipeline,
        )
        from diffusers import AutoencoderKL
        from diffusers.models import HiDreamImageTransformer2DModel
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5Tokenizer,
            LlamaForCausalLM,
            PreTrainedTokenizerFast,
        )

        model_path = self.CONFIG.base_model

        logger.info("[1/4] Loading Llama 8B")
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_PATH, use_fast=False, token=self.hf_token)
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_PATH,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=self.dtype,
            token=self.hf_token,
        ).to(self.device)
        text_encoder_4.eval().requires_grad_(False)

        logger.info("[2/4] Loading transformer")
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
            token=self.hf_token,
        ).to(self.device)
        transformer.eval().requires_grad_(False)

        logger.info("[3/4] Loading VAE and text encoders")
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        vae.eval().requires_grad_(False)

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", token=self.hf_token)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", token=self.hf_token)
        text_encoder_3 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_3", torch_dtype=self.dtype, token=self.hf_token
        ).to(self.device)
        tokenizer_3 = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer_3", token=self.hf_token)

        for te in [text_encoder, text_encoder_2, text_encoder_3]:
            te.eval().requires_grad_(False)

        logger.info("[4/4] Creating pipeline")
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)

        self.pipe = HiDreamImageEditingPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
            text_encoder_4=text_encoder_4,
            tokenizer_4=tokenizer_4,
            transformer=transformer,
        )
        self.transformer = transformer
        self.vae = vae

        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info("HiDream-E1 pipeline loaded")

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """Load LoRA using fused mode."""
        if not lora_paths:
            return

        lora_path = lora_paths[0]
        if isinstance(lora_path, dict):
            lora_path = list(lora_path.values())[0] if lora_path else None
        if not lora_path or not os.path.exists(lora_path):
            logger.warning(f"LoRA not found: {lora_path}")
            return

        logger.info(f"Loading LoRA (fused): {lora_path}")
        lora_sd = load_file(lora_path)
        converted = {k.replace("diffusion_model.", ""): v for k, v in lora_sd.items()}

        pairs = {}
        for k, v in converted.items():
            if "lora_A" in k:
                base = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                pairs.setdefault(base, {})["A"] = v
            elif "lora_B" in k:
                base = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                pairs.setdefault(base, {})["B"] = v

        model_sd = self.transformer.state_dict()
        updated = 0
        for base, pair in pairs.items():
            if "A" in pair and "B" in pair:
                key = f"{base}.weight"
                if key in model_sd:
                    delta = lora_scale * (pair["B"] @ pair["A"])
                    model_sd[key] = model_sd[key] + delta.to(model_sd[key].device, dtype=model_sd[key].dtype)
                    updated += 1

        if updated:
            self.transformer.load_state_dict(model_sd)
            self.lora_loaded = True
            self._current_lora_scale = lora_scale
            logger.info(f"Fused {updated} LoRA pairs, scale={lora_scale}")

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
        if control_image is None:
            raise ValueError("HiDream-E1 requires control_image")

        # Resize if needed
        if control_image.size != (width, height):
            control_image = control_image.resize((width, height), Image.BILINEAR)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                image=control_image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=4.0,
                generator=generator,
                max_sequence_length=128,
            )
        return {"image": result.images[0]}

    def unload(self):
        for attr in ["pipe", "transformer", "vae"]:
            if getattr(self, attr, None):
                delattr(self, attr)
        self.lora_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HiDream-E1 unloaded")
