"""
Chroma pipeline implementation.

Note: Chroma requires ai-toolkit imports for custom pipeline and model classes.
"""

import os
import sys
import gc
import logging
from typing import Dict, Any, Optional

import torch
from PIL import Image

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType
from ..config import settings

logger = logging.getLogger(__name__)

# Scheduler config from ai-toolkit
CHROMA_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}

# Quantization config - aligned with AI Toolkit training (float8)
QUANTIZE_TRANSFORMER = True
QUANTIZE_TE = True
QTYPE = "float8"  # torchao float8 quantization


class ChromaPipeline(BasePipeline):
    """
    Chroma pipeline.

    Uses ai-toolkit's custom ChromaPipeline and Chroma transformer.

    Key alignment points:
    - Custom model loading from ai-toolkit
    - LoRA merged before quantization
    - FakeCLIP for text encoder compatibility
    - Resolution divisor: 32
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.CHROMA,
        base_model="lodestones/Chroma1-Base",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        lora_merge_method=LoraMergeMethod.CUSTOM,  # Manual LoRA merge before quantization
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder_list = None
        self.tokenizer_list = None
        self.transformer = None
        self._lora_path = None
        self._lora_scale = 1.0

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Load the Chroma pipeline with LoRA merged before quantization."""
        # Extract first LoRA path (string or dict MoE format)
        if lora_paths:
            first_lora = lora_paths[0]
            self._lora_path = list(first_lora.values())[0] if isinstance(first_lora, dict) else first_lora

        self._lora_scale = lora_scale

        logger.info(f"Loading Chroma with lora_path={self._lora_path}, lora_scale={self._lora_scale}")
        self._load_pipeline()

    def _load_pipeline(self):
        # Add ai-toolkit to path (configurable via AI_TOOLKIT_PATH env var)
        ai_toolkit_path = settings.ai_toolkit_path
        if os.path.exists(ai_toolkit_path) and ai_toolkit_path not in sys.path:
            sys.path.insert(0, ai_toolkit_path)

        try:
            from diffusers import AutoencoderKL
            from transformers import T5TokenizerFast, T5EncoderModel
            from safetensors.torch import load_file
            import huggingface_hub

            # Import ai-toolkit components
            from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
            from toolkit.util.quantize import quantize, get_qtype
            from toolkit.basic import flush
            from extensions_built_in.diffusion_models.chroma.pipeline import ChromaPipeline as AITKChromaPipeline
            from extensions_built_in.diffusion_models.chroma.src.model import Chroma, chroma_params

            model_path = self.CONFIG.base_model
            extras_path = "ostris/Flex.1-alpha"  # T5 and VAE source

            # Resolve model path
            if model_path.startswith("lodestones/Chroma1-"):
                model_path = huggingface_hub.hf_hub_download(
                    repo_id=model_path,
                    filename=f"{model_path.split('/')[-1]}.safetensors",
                    token=self.hf_token,
                )

            # ========== Step 1: Load transformer ==========
            logger.info("[1/6] Loading Chroma transformer")
            chroma_state_dict = load_file(model_path, "cpu")

            # Determine block counts
            double_blocks = 0
            single_blocks = 0
            for key in chroma_state_dict.keys():
                if "double_blocks" in key:
                    block_num = int(key.split(".")[1]) + 1
                    double_blocks = max(double_blocks, block_num)
                elif "single_blocks" in key:
                    block_num = int(key.split(".")[1]) + 1
                    single_blocks = max(single_blocks, block_num)

            logger.info(f"Double Blocks: {double_blocks}, Single Blocks: {single_blocks}")

            chroma_params.depth = double_blocks
            chroma_params.depth_single_blocks = single_blocks
            transformer = Chroma(chroma_params)
            transformer.dtype = self.dtype
            transformer.load_state_dict(chroma_state_dict)

            # Set FakeConfig
            class FakeConfig:
                def __init__(self):
                    self.attention_head_dim = 128
                    self.guidance_embeds = True
                    self.in_channels = 64
                    self.joint_attention_dim = 4096
                    self.num_attention_heads = 24
                    self.num_layers = double_blocks
                    self.num_single_layers = single_blocks
                    self.patch_size = 1

            transformer.config = FakeConfig()
            transformer.to("cpu", dtype=self.dtype)

            # ========== Step 1.5: Load and merge LoRA (before quantization) ==========
            if self._lora_path and os.path.exists(self._lora_path):
                logger.info(
                    f"[1.5/6] Loading LoRA (BEFORE quantization): {self._lora_path} with scale={self._lora_scale}"
                )
                self._merge_lora_to_transformer(transformer, self._lora_path, self._lora_scale)
                self.lora_loaded = True
            elif self._lora_path:
                logger.warning(f"LoRA file not found: {self._lora_path}")

            # ========== Step 2: Quantize transformer with float8 ==========
            if QUANTIZE_TRANSFORMER:
                logger.info(f"Quantizing transformer with {QTYPE}...")
                quantization_type = get_qtype(QTYPE)
                # float8 (torchao): move to GPU first, then quantize
                transformer.to(self.device, dtype=self.dtype)
                quantize(transformer, weights=quantization_type)
                logger.info("Transformer quantized")
            else:
                transformer.to(self.device, dtype=self.dtype)

            flush()

            # ========== Step 3: Load T5 ==========
            logger.info("[2/6] Loading T5 text encoder")
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                extras_path,
                subfolder="tokenizer_2",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                extras_path,
                subfolder="text_encoder_2",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )

            # Quantize T5 with float8
            if QUANTIZE_TE:
                logger.info(f"Quantizing T5 with {QTYPE}...")
                # float8 (torchao): move to GPU first, then quantize
                text_encoder_2.to(self.device, dtype=self.dtype)
                quantize(text_encoder_2, weights=get_qtype(QTYPE))
                logger.info("T5 quantized")
            else:
                text_encoder_2.to(self.device)

            flush()

            # ========== Step 4: Create FakeCLIP ==========
            logger.info("[3/6] Creating FakeCLIP")

            class FakeCLIP(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.dtype = torch.bfloat16
                    self.device = "cuda"
                    self.text_model = None
                    self.tokenizer = None
                    self.model_max_length = 77

                def forward(self, *args, **kwargs):
                    return torch.zeros(1, 1, 1).to(self.device)

            text_encoder = FakeCLIP()
            tokenizer = FakeCLIP()
            text_encoder.to(self.device, dtype=self.dtype)

            # ========== Step 5: Create scheduler ==========
            logger.info("[4/6] Creating scheduler")
            scheduler = CustomFlowMatchEulerDiscreteScheduler(**CHROMA_SCHEDULER_CONFIG)

            # ========== Step 6: Load VAE ==========
            logger.info("[5/6] Loading VAE")
            vae = AutoencoderKL.from_pretrained(
                extras_path,
                subfolder="vae",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
            vae.to(self.device, dtype=self.dtype)

            # ========== Step 7: Create pipeline ==========
            logger.info("[6/6] Creating Chroma pipeline")
            self.pipe = AITKChromaPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=None,
                tokenizer_2=tokenizer_2,
                vae=vae,
                transformer=None,
            )
            self.pipe.text_encoder_2 = text_encoder_2
            self.pipe.transformer = transformer

            self.text_encoder_list = [text_encoder, text_encoder_2]
            self.tokenizer_list = [tokenizer, tokenizer_2]
            self.transformer = transformer

            # Set eval mode and disable gradients
            transformer.requires_grad_(False)
            transformer.eval()
            text_encoder.to(self.device)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            text_encoder_2.requires_grad_(False)
            text_encoder_2.eval()
            vae.requires_grad_(False)
            vae.eval()

            flush()

            # Print GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

            logger.info("Chroma pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Chroma pipeline: {e}")
            raise

    def _merge_lora_to_transformer(self, transformer, lora_path: str, lora_scale: float = 1.0):
        """Merge LoRA weights into transformer - aligned with chroma_inference.py"""
        from safetensors.torch import load_file

        lora_state_dict = load_file(lora_path, device="cpu")

        # Remove diffusion_model. prefix (same as chroma_inference.py)
        converted_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "")
            converted_sd[new_key] = value.to(self.dtype)

        # Get LoRA rank
        rank = None
        for k in converted_sd:
            if "lora_A.weight" in k:
                rank = converted_sd[k].shape[0]
                break

        if not rank:
            logger.warning("No lora_A.weight found in LoRA")
            return

        logger.info(f"LoRA rank: {rank}")

        # Merge LoRA weights (same logic as chroma_inference.py)
        transformer_state = transformer.state_dict()
        merged_count = 0

        for key in list(converted_sd.keys()):
            if "lora_A.weight" in key:
                base_key = key.replace(".lora_A.weight", "")
                lora_a_key = key
                lora_b_key = key.replace("lora_A.weight", "lora_B.weight")

                if lora_b_key in converted_sd:
                    weight_key = base_key + ".weight"

                    if weight_key in transformer_state:
                        lora_a = converted_sd[lora_a_key]
                        lora_b = converted_sd[lora_b_key]

                        # LoRA: W' = W + B @ A
                        delta = lora_b @ lora_a

                        # Get alpha
                        alpha_key = base_key + ".alpha"
                        if alpha_key in converted_sd:
                            alpha = converted_sd[alpha_key].item()
                        else:
                            alpha = rank

                        # Apply both internal scale (alpha/rank) and external lora_scale
                        scale = (alpha / rank) * lora_scale

                        original_weight = transformer_state[weight_key]
                        transformer_state[weight_key] = (
                            original_weight
                            + delta.to(device=original_weight.device, dtype=original_weight.dtype) * scale
                        )
                        merged_count += 1

        transformer.load_state_dict(transformer_state)
        logger.info(f"Merged {merged_count} LoRA layers with scale={lora_scale}")

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """
        For Chroma, LoRA is already merged in _load_pipeline() during transformer loading.
        This method should not be called directly.
        """
        if self.lora_loaded:
            logger.info("LoRA already loaded, skipping _load_lora")
            return
        logger.warning("_load_lora called but LoRA should be merged in _load_pipeline()")

    def _get_prompt_embeds(self, prompt: str):
        """Get prompt embeddings using T5."""
        from toolkit.prompt_utils import PromptEmbeds

        max_length = 512
        te_device = self.text_encoder_list[1].device
        te_dtype = self.text_encoder_list[1].dtype

        text_inputs = self.tokenizer_list[1](
            [prompt],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder_list[1](text_input_ids.to(te_device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=te_dtype, device=te_device)
        prompt_attention_mask = text_inputs["attention_mask"].to(te_device)

        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_attention_mask
        return pe

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
        """Run Chroma inference - aligned with chroma_inference.py"""
        # Get embeddings (same as chroma_inference.py get_prompt_embeds)
        conditional_embeds = self._get_prompt_embeds(prompt)
        unconditional_embeds = self._get_prompt_embeds("")

        extra = {
            "negative_prompt_embeds": unconditional_embeds.text_embeds,
            "negative_prompt_attn_mask": unconditional_embeds.attention_mask,
        }

        # Aligned with chroma_inference.py: use torch.no_grad() + autocast
        # Note: base.py already wraps with torch.inference_mode(), but we keep
        # torch.no_grad() for consistency with the script
        with torch.cuda.amp.autocast(dtype=self.dtype):
            result = self.pipe(
                prompt_embeds=conditional_embeds.text_embeds,
                prompt_attn_mask=conditional_embeds.attention_mask,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=None,  # Aligned with script (default None)
                generator=generator,
                **extra,
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
        if self.text_encoder_list is not None:
            del self.text_encoder_list
            self.text_encoder_list = None
        if self.tokenizer_list is not None:
            del self.tokenizer_list
            self.tokenizer_list = None

        self.lora_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Chroma pipeline unloaded")
