"""
FLUX.2 pipeline implementation with LoRA hotswap support.

Note: FLUX.2 requires ai-toolkit for custom pipeline and model classes.
This version adds minimal changes for LoRA switching while keeping
the exact same inference behavior as the original.
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

# Environment variable to enable aggressive memory cleanup (default: off for faster iterative runs)
AITK_AGGRESSIVE_CLEANUP = os.environ.get("AITK_AGGRESSIVE_CLEANUP", "").lower() in ("1", "true", "yes")
# Opt-in: keep a full CPU snapshot of transformer weights for fast LoRA switching.
# Default off to avoid duplicating RAM.
AITK_FLUX2_HOTSWAP = os.environ.get("AITK_FLUX2_HOTSWAP", "").lower() in ("1", "true", "yes")


def _maybe_cleanup() -> None:
    """Optionally run aggressive cleanup (gc + CUDA cache clear)."""
    if not AITK_AGGRESSIVE_CLEANUP:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Scheduler config from ai-toolkit
FLUX2_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


class Flux2Pipeline(BasePipeline):
    """
    FLUX.2 pipeline with LoRA hotswap support.

    Uses ai-toolkit's custom Flux2Pipeline and Flux2 transformer.

    Key alignment points:
    - Custom model loading from ai-toolkit
    - LoRA merged before moving to GPU
    - Mistral text encoder with quantization
    - Resolution divisor: 32
    - Original weights saved for fast LoRA switching
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLUX2,
        base_model="black-forest-labs/FLUX.2-dev",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        lora_merge_method=LoraMergeMethod.CUSTOM,  # Manual LoRA merge with hotswap
    )

    # Model/pipeline variants (overridden by Klein subclasses)
    TRANSFORMER_FILENAME = "flux2-dev.safetensors"
    TEXT_ENCODER_REPO = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    TEXT_ENCODER_TYPE = "mistral"  # "mistral" or "qwen"
    VAE_REPO = None  # Optional separate VAE repo (defaults to base model)
    IS_GUIDANCE_DISTILLED = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self._lora_path = None
        self._lora_scale = 1.0
        # Store original transformer weights (on CPU) for fast LoRA switching
        self._original_transformer_state: Optional[Dict[str, torch.Tensor]] = None
        # Timing details for pipeline loading
        self.timings: Dict[str, float] = {}

    def _get_flux2_params(self):
        from extensions_built_in.diffusion_models.flux2.src.model import Flux2Params
        return Flux2Params()

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Load the FLUX.2 pipeline with LoRA merged."""
        logger.info(f"Flux2Pipeline.load() called with lora_paths={lora_paths}, lora_scale={lora_scale}")

        # Extract first LoRA path (string or dict MoE format)
        if lora_paths:
            first_lora = lora_paths[0]
            self._lora_path = list(first_lora.values())[0] if isinstance(first_lora, dict) else first_lora

        self._lora_scale = lora_scale

        logger.info(f"Loading FLUX.2 with lora_path={self._lora_path}, lora_scale={self._lora_scale}")
        self._load_pipeline()

    def _load_pipeline(self):
        """Load FLUX.2 pipeline using ai-toolkit components."""
        import time

        logger.info(
            f">>> _load_pipeline() START: self._lora_path={self._lora_path}, self._lora_scale={self._lora_scale}"
        )

        # Reset timings
        self.timings = {}

        # Add ai-toolkit to path (configurable via AI_TOOLKIT_PATH env var)
        ai_toolkit_path = settings.ai_toolkit_path
        if os.path.exists(ai_toolkit_path) and ai_toolkit_path not in sys.path:
            sys.path.insert(0, ai_toolkit_path)

        try:
            from safetensors.torch import load_file
            import huggingface_hub

            # Try to import ai-toolkit components
            try:
                from extensions_built_in.diffusion_models.flux2.src.pipeline import Flux2Pipeline as AITKFlux2Pipeline
                from extensions_built_in.diffusion_models.flux2.src.model import Flux2
                from extensions_built_in.diffusion_models.flux2.src.autoencoder import AutoEncoder, AutoEncoderParams
                from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
                from toolkit.util.quantize import quantize, get_qtype
                from optimum.quanto import freeze
            except ImportError:
                raise ImportError(
                    f"FLUX.2 requires ai-toolkit. Please ensure AI_TOOLKIT_PATH ({ai_toolkit_path}) is valid."
                )

            # Clear GPU memory before loading
            _maybe_cleanup()

            base_model_path = self.CONFIG.base_model

            # 1. Load Transformer
            t_start = time.perf_counter()
            logger.info("Loading FLUX.2 Transformer")
            with torch.device("meta"):
                transformer = Flux2(self._get_flux2_params())

            # Download transformer weights
            transformer_filename = self.TRANSFORMER_FILENAME
            transformer_path = huggingface_hub.hf_hub_download(
                repo_id=base_model_path,
                filename=transformer_filename,
                token=self.hf_token,
            )

            state_dict = load_file(transformer_path, device="cpu")
            for key in state_dict:
                state_dict[key] = state_dict[key].to(self.dtype)
            transformer.load_state_dict(state_dict, assign=True)
            self.timings["load_transformer"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] load_transformer: {self.timings['load_transformer']:.3f}s")

            # Optional: Save original weights before LoRA merge (RAM-heavy).
            if AITK_FLUX2_HOTSWAP:
                t_start = time.perf_counter()
                logger.info("Saving original transformer weights for LoRA switching (AITK_FLUX2_HOTSWAP=1)")
                self._original_transformer_state = {k: v.clone() for k, v in transformer.state_dict().items()}
                self.timings["save_original_weights"] = time.perf_counter() - t_start
                logger.info(f"Saved {len(self._original_transformer_state)} original weight tensors")
                logger.info(f"[TIMING] save_original_weights: {self.timings['save_original_weights']:.3f}s")
            else:
                self._original_transformer_state = None
                logger.info(
                    "Skipping original transformer weight snapshot (AITK_FLUX2_HOTSWAP not enabled); "
                    "LoRA switching will require full reload."
                )

            # 2. Load and merge LoRA before moving to GPU
            t_start = time.perf_counter()
            logger.info(
                f"LoRA check: lora_path={self._lora_path}, exists={os.path.exists(self._lora_path) if self._lora_path else 'N/A'}"
            )
            if self._lora_path and os.path.exists(self._lora_path):
                logger.info(f"Loading and merging LoRA: {self._lora_path} with scale={self._lora_scale}")
                self._merge_lora_to_transformer(transformer, self._lora_path, self._lora_scale)
            elif self._lora_path:
                logger.error(f"LoRA file not found: {self._lora_path}")
            else:
                logger.warning("No LoRA path provided, using base model only")
            self.timings["merge_lora"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] merge_lora: {self.timings['merge_lora']:.3f}s")

            t_start = time.perf_counter()
            transformer.to(self.device, dtype=self.dtype)
            transformer.eval()
            self.timings["transformer_to_gpu"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] transformer_to_gpu: {self.timings['transformer_to_gpu']:.3f}s")

            # Clear memory after transformer load
            _maybe_cleanup()

            # 3. Load VAE
            t_start = time.perf_counter()
            logger.info("Loading FLUX.2 VAE")
            with torch.device("meta"):
                vae = AutoEncoder(AutoEncoderParams())

            vae_filename = "ae.safetensors"
            vae_repo = self.VAE_REPO or base_model_path
            vae_path = huggingface_hub.hf_hub_download(
                repo_id=vae_repo,
                filename=vae_filename,
                token=self.hf_token,
            )

            vae_state_dict = load_file(vae_path, device="cpu")
            for key in vae_state_dict:
                vae_state_dict[key] = vae_state_dict[key].to(self.dtype)
            vae.load_state_dict(vae_state_dict, assign=True)
            vae.to(self.device, dtype=self.dtype)
            vae.eval()
            self.timings["load_vae"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] load_vae: {self.timings['load_vae']:.3f}s")

            # Clear memory after VAE load
            _maybe_cleanup()

            # 4. Load Text Encoder
            t_start = time.perf_counter()
            if self.TEXT_ENCODER_TYPE == "mistral":
                from transformers import AutoProcessor, Mistral3ForConditionalGeneration

                logger.info(f"Loading Mistral Text Encoder: {self.TEXT_ENCODER_REPO}")
                text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                    self.TEXT_ENCODER_REPO,
                    torch_dtype=self.dtype,
                    token=self.hf_token,
                )
            elif self.TEXT_ENCODER_TYPE == "qwen":
                from transformers import Qwen3ForCausalLM

                logger.info(f"Loading Qwen3 Text Encoder: {self.TEXT_ENCODER_REPO}")
                text_encoder = Qwen3ForCausalLM.from_pretrained(
                    self.TEXT_ENCODER_REPO,
                    torch_dtype=self.dtype,
                    token=self.hf_token,
                )
            else:
                raise ValueError(f"Unsupported text encoder type: {self.TEXT_ENCODER_TYPE}")
            self.timings["load_text_encoder"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] load_text_encoder: {self.timings['load_text_encoder']:.3f}s")

            # 5. Apply qfloat8 quantization (optional, currently disabled)
            t_start = time.perf_counter()

            # logger.info("Applying qfloat8 quantization to text encoder")
            # qtype = get_qtype("qfloat8")
            # quantize(text_encoder, weights=qtype)
            # freeze(text_encoder)

            text_encoder.to(self.device)
            text_encoder.eval()
            self.timings["text_encoder"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] text_encoder: {self.timings['text_encoder']:.3f}s")

            # 6. Load Tokenizer
            if self.TEXT_ENCODER_TYPE == "mistral":
                from transformers import AutoProcessor

                tokenizer = AutoProcessor.from_pretrained(
                    self.TEXT_ENCODER_REPO,
                    token=self.hf_token,
                )
            elif self.TEXT_ENCODER_TYPE == "qwen":
                from transformers import Qwen2Tokenizer

                tokenizer = Qwen2Tokenizer.from_pretrained(
                    self.TEXT_ENCODER_REPO,
                    token=self.hf_token,
                )
            else:
                raise ValueError(f"Unsupported text encoder type: {self.TEXT_ENCODER_TYPE}")

            # 7. Create Scheduler
            scheduler = CustomFlowMatchEulerDiscreteScheduler(**FLUX2_SCHEDULER_CONFIG)

            # 8. Create Pipeline
            logger.info("Creating FLUX.2 Pipeline")
            self.pipe = AITKFlux2Pipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                text_encoder_type=self.TEXT_ENCODER_TYPE,
                is_guidance_distilled=self.IS_GUIDANCE_DISTILLED,
            )

            self.transformer = transformer
            self.vae = vae
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer

            _maybe_cleanup()

            logger.info("FLUX.2 pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load FLUX.2 pipeline: {e}")
            raise

    def _merge_lora_to_transformer(self, transformer, lora_path: str, lora_scale: float = 1.0):
        """Merge LoRA weights into transformer (GPU addmm)."""
        from safetensors.torch import load_file

        # Ensure merge happens on GPU when available
        try:
            param_device = next(transformer.parameters()).device
        except StopIteration:
            param_device = torch.device("cpu")
        if param_device.type != "cuda":
            logger.info("Moving transformer to GPU for LoRA merge")
            transformer.to(self.device, dtype=self.dtype)
            transformer.eval()
            try:
                param_device = next(transformer.parameters()).device
            except StopIteration:
                param_device = torch.device("cpu")
        logger.info(f"LoRA merge device: {param_device}")
        # TODO: If OOMs appear in smaller GPUs, consider chunked merge or reintroduce periodic cache clears behind a flag.

        lora_state_dict = load_file(lora_path, device="cpu")
        logger.info(f"Loaded LoRA file with {len(lora_state_dict)} keys")

        # Log first few keys from LoRA file
        lora_keys_sample = list(lora_state_dict.keys())[:5]
        logger.info(f"Sample LoRA keys (before conversion): {lora_keys_sample}")

        # Remove diffusion_model. prefix
        converted_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "")
            converted_sd[new_key] = value.to(self.dtype)

        # Move LoRA tensors to merge device once (avoid per-layer CPU->GPU transfers)
        if param_device.type == "cuda":
            for key, value in converted_sd.items():
                if isinstance(value, torch.Tensor) and value.device.type != "cuda":
                    converted_sd[key] = value.to(device=param_device, dtype=self.dtype, non_blocking=True)

        # Log first few keys after conversion
        converted_keys_sample = list(converted_sd.keys())[:5]
        logger.info(f"Sample LoRA keys (after conversion): {converted_keys_sample}")

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

        # Log sample transformer keys
        transformer_keys_sample = list(transformer.state_dict().keys())[:5]
        logger.info(f"Sample transformer keys: {transformer_keys_sample}")

        # Manual merge (GPU-friendly): W += scale * (B @ A)
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

                        alpha_key = base_key + ".alpha"
                        alpha_val = converted_sd.get(alpha_key, None)
                        if alpha_val is None:
                            alpha = rank
                        else:
                            alpha = alpha_val.item() if isinstance(alpha_val, torch.Tensor) else float(alpha_val)

                        # Apply both internal scale (alpha/rank) and external lora_scale
                        scale = (alpha / rank) * lora_scale

                        original_weight = transformer_state[weight_key]
                        device = original_weight.device
                        dtype = original_weight.dtype
                        lora_a = lora_a.to(device=device, dtype=dtype, non_blocking=True)
                        lora_b = lora_b.to(device=device, dtype=dtype, non_blocking=True)

                        # In-place: W = W + scale * (B @ A)
                        original_weight.addmm_(lora_b, lora_a, beta=1.0, alpha=scale)
                        merged_count += 1


        transformer.load_state_dict(transformer_state, assign=True)
        logger.info(f"Merged {merged_count} LoRA layers with scale={lora_scale}")

    def _restore_original_weights(self):
        """Restore transformer to original weights (before any LoRA merge)."""
        if self._original_transformer_state is None:
            logger.warning("No original weights saved")
            return

        # Get current state dict and restore from CPU cache
        state_dict = self.transformer.state_dict()
        for key in state_dict:
            if key in self._original_transformer_state:
                # copy_() handles cross-device transfer internally without extra allocation
                state_dict[key].copy_(self._original_transformer_state[key])

        _maybe_cleanup()

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """LoRA is already merged in _load_pipeline for FLUX.2."""
        pass  # Already handled

    def set_lora_scale(self, scale: float) -> bool:
        """
        Dynamically set LoRA scale by restoring original weights and re-merging.

        Args:
            scale: New LoRA scale value

        Returns:
            True if successful, False otherwise
        """
        import time

        if scale == self._lora_scale:
            return True

        if self._original_transformer_state is None:
            logger.warning("Original weights not saved, cannot change scale")
            return False

        if not self._lora_path:
            logger.warning("No LoRA loaded, cannot change scale")
            return False

        try:
            total_start = time.perf_counter()
            logger.info(f"=== set_lora_scale: {self._lora_scale} -> {scale} ===")

            # Restore original weights
            restore_start = time.perf_counter()
            self._restore_original_weights()
            restore_elapsed = time.perf_counter() - restore_start
            logger.info(f"[TIMING] restore_original_weights: {restore_elapsed:.3f}s")

            # Re-merge with new scale
            merge_start = time.perf_counter()
            self._merge_lora_to_transformer(self.transformer, self._lora_path, scale)
            merge_elapsed = time.perf_counter() - merge_start
            logger.info(f"[TIMING] merge_lora: {merge_elapsed:.3f}s")

            self._lora_scale = scale
            total_elapsed = time.perf_counter() - total_start
            logger.info(f"[TIMING] set_lora_scale TOTAL: {total_elapsed:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to change LoRA scale: {e}")
            return False

    def switch_lora(self, lora_paths: list, lora_scale: float = 1.0) -> bool:
        """
        Switch to a new LoRA without reloading the base model.

        Args:
            lora_paths: New LoRA paths to load
            lora_scale: LoRA strength

        Returns:
            True if successful, False otherwise
        """
        import time

        if self.pipe is None or self.transformer is None:
            logger.warning("Pipeline not loaded, cannot switch LoRA")
            return False

        if self._original_transformer_state is None:
            logger.warning("Original weights not saved, cannot switch LoRA")
            return False

        if not lora_paths:
            logger.warning("No LoRA paths provided for switching")
            return False

        first_lora = lora_paths[0]
        new_lora_path = list(first_lora.values())[0] if isinstance(first_lora, dict) else first_lora

        if not os.path.exists(new_lora_path):
            logger.error(f"New LoRA file not found: {new_lora_path}")
            return False

        try:
            total_start = time.perf_counter()
            logger.info(f"=== switch_lora: {self._lora_path} -> {new_lora_path} ===")

            # Restore original weights
            restore_start = time.perf_counter()
            self._restore_original_weights()
            restore_elapsed = time.perf_counter() - restore_start
            logger.info(f"[TIMING] restore_original_weights: {restore_elapsed:.3f}s")

            # Merge new LoRA
            merge_start = time.perf_counter()
            self._merge_lora_to_transformer(self.transformer, new_lora_path, lora_scale)
            merge_elapsed = time.perf_counter() - merge_start
            logger.info(f"[TIMING] merge_lora: {merge_elapsed:.3f}s")

            self._lora_path = new_lora_path
            self._lora_scale = lora_scale
            total_elapsed = time.perf_counter() - total_start
            logger.info(f"[TIMING] switch_lora TOTAL: {total_elapsed:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to switch LoRA: {e}")
            return False

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
        """Run FLUX.2 inference (exact same as original)."""
        negative_prompt = negative_prompt or ""
        if self.IS_GUIDANCE_DISTILLED and negative_prompt:
            logger.warning("FLUX.2 is guidance-distilled; negative_prompt will be ignored")

        # Build control_img_list from control_image/control_images
        control_img_list = []
        if control_image is not None:
            # Ensure RGB mode for ai-toolkit compatibility
            if control_image.mode != "RGB":
                control_image = control_image.convert("RGB")
            control_img_list.append(control_image)
        if control_images:
            for img in control_images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                control_img_list.append(img)

        # Keep exact same call signature as original
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            control_img_list=control_img_list if control_img_list else None,
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
        if self.vae is not None:
            del self.vae
            self.vae = None
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self._original_transformer_state is not None:
            del self._original_transformer_state
            self._original_transformer_state = None

        self.lora_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("FLUX.2 pipeline unloaded")
