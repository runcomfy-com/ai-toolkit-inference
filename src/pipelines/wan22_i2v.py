"""
Wan 2.2 I2V 14B pipeline implementation.
"""

import os
import sys
import logging
import time
from typing import Dict, Any, Optional, Union

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from .torch_inductor_config import apply_torch_inductor_optimizations
from ..schemas.models import ModelType
from ..config import settings

logger = logging.getLogger(__name__)

apply_torch_inductor_optimizations()

# Scheduler config (from wan22_14b_model.py scheduler_configUniPC)
# 14B: flow_shift=3.0 (5B is 5.0)
WAN22_14B_I2V_SCHEDULER_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,  # 14B: 3.0, 5B: 5.0
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}

# I2V specific: boundary_ratio=0.9 (T2V uses 0.875)
BOUNDARY_RATIO = 0.9


class Wan22I2V14BPipeline(BasePipeline):
    """
    Wan 2.2 I2V 14B pipeline using AI Toolkit's Wan22Pipeline.

    Key alignment points:
    - Uses AI Toolkit's Wan22Pipeline (exact match with training sample behavior)
    - Dual Transformer architecture (MoE): high_noise -> transformer, low_noise -> transformer_2
    - expand_timesteps=False (14B specific)
    - boundary_ratio=0.9 (I2V specific)
    - flow_shift=3.0 (14B specific)
    - First frame conditioning via add_first_frame_conditioning
    - Resolution divisor: 16 (8x compression + 2x2 patch)
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.WAN22_14B_I2V,
        base_model="ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
        resolution_divisor=16,  # 14B: 8x compression + 2x2 patch = 16
        default_steps=25,
        default_guidance_scale=4.0,
        requires_control_image=True,
        is_video_model=True,
        default_num_frames=41,
        default_fps=16,
        enable_cpu_offload=False,  # Disabled due to device compatibility with Wan22Pipeline
        enable_xformers=True,  # Enable xformers memory efficient attention
    )

    # Model paths (aligned with AI Toolkit training)
    VAE_MODEL = "ai-toolkit/wan2.1-vae"
    TEXT_ENCODER_MODEL = "ai-toolkit/umt5_xxl_encoder"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lora_high_path = None
        self._lora_low_path = None

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Load the Wan22 I2V pipeline with LoRA.

        LoRA format for Wan 2.2 14B MoE: [{"high": "path", "low": "path"}]
        """
        # Extract LoRA paths from MoE format
        if lora_paths and isinstance(lora_paths[0], dict):
            lora_config = lora_paths[0]
            self._lora_high_path = lora_config.get("high")
            self._lora_low_path = lora_config.get("low")

        logger.info(f"Loading Wan22 I2V with high_noise={self._lora_high_path}, low_noise={self._lora_low_path}")
        self._load_pipeline()

        # Enable xformers for both transformers
        self._enable_xformers()

        # Load LoRA if paths provided
        if self._lora_high_path or self._lora_low_path:
            self._load_lora(lora_paths, lora_scale)

    def _load_pipeline(self):
        """Load Wan 2.2 I2V 14B pipeline using AI Toolkit components."""
        # Add ai-toolkit to path (configurable via AI_TOOLKIT_PATH env var)
        ai_toolkit_path = settings.ai_toolkit_path
        if os.path.exists(ai_toolkit_path) and ai_toolkit_path not in sys.path:
            sys.path.insert(0, ai_toolkit_path)

        try:
            from diffusers import UniPCMultistepScheduler
            from diffusers import WanTransformer3DModel, AutoencoderKLWan
            from transformers import AutoTokenizer, UMT5EncoderModel

            # Use AI Toolkit's custom Pipeline
            from extensions_built_in.diffusion_models.wan22.wan22_pipeline import Wan22Pipeline

            logger.info("Loading model components:")
            logger.info(f"  Transformer: {self.CONFIG.base_model}")
            logger.info(f"  VAE: {self.VAE_MODEL}")
            logger.info(f"  Text Encoder: {self.TEXT_ENCODER_MODEL}")

            # Load VAE
            logger.info("  Loading VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                self.VAE_MODEL,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )

            # Load Transformer 1 (high noise)
            logger.info("  Loading Transformer 1 (high noise)...")
            transformer = WanTransformer3DModel.from_pretrained(
                self.CONFIG.base_model,
                subfolder="transformer",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )

            # Load Transformer 2 (low noise)
            logger.info("  Loading Transformer 2 (low noise)...")
            transformer_2 = WanTransformer3DModel.from_pretrained(
                self.CONFIG.base_model,
                subfolder="transformer_2",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )

            # Load Text Encoder
            logger.info("  Loading Text Encoder...")
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.TEXT_ENCODER_MODEL,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                token=self.hf_token,
            )

            # Load Tokenizer
            logger.info("  Loading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.TEXT_ENCODER_MODEL,
                subfolder="tokenizer",
                token=self.hf_token,
            )

            # Create Scheduler
            logger.info(f"  Creating Scheduler (flow_shift={WAN22_14B_I2V_SCHEDULER_CONFIG['flow_shift']})...")
            scheduler = UniPCMultistepScheduler(**WAN22_14B_I2V_SCHEDULER_CONFIG)

            # Create Wan22Pipeline (AI Toolkit custom)
            logger.info(f"  Creating Wan22Pipeline (expand_timesteps=False, boundary_ratio={BOUNDARY_RATIO})...")
            self.pipe = Wan22Pipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                transformer_2=transformer_2,
                scheduler=scheduler,
                expand_timesteps=False,  # 14B: False, 5B: True
                boundary_ratio=BOUNDARY_RATIO,  # I2V: 0.9, T2V: 0.875
                device=torch.device(self.device),
                aggressive_offload=False,
            )

            # Move all components to GPU (no CPU offload for this model)
            logger.info(f"  Moving components to {self.device}...")
            self.pipe.vae.to(self.device)
            self.pipe.text_encoder.to(self.device)
            self.pipe.transformer.to(self.device)
            self.pipe.transformer_2.to(self.device)

            logger.info("  Pipeline created successfully")

        except Exception as e:
            logger.error(f"Failed to load Wan22 I2V pipeline: {e}")
            raise

    def _enable_xformers(self):
        """
        Enable xformers memory efficient attention for both transformers.

        Override base method because Wan22Pipeline uses dual transformer architecture.
        """
        if self.pipe is None:
            return

        if not self.CONFIG.enable_xformers:
            logger.debug(f"xformers disabled for {self.__class__.__name__}")
            return

        try:
            self.pipe.transformer.enable_xformers_memory_efficient_attention()
            self.pipe.transformer_2.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled for both transformers")
        except Exception as e:
            logger.debug(f"xformers not available, using default attention: {e}")

    def _apply_torch_compile(self):
        """
        Apply torch.compile to transformer models for inference acceleration.

        Must be called AFTER LoRA loading to ensure compatibility with set_adapters.
        Uses dynamic=True to allow adapter weight changes without recompilation.
        First inference will be slow due to compilation, subsequent calls are faster.
        """
        if getattr(self, "_compiled", False):
            logger.info("torch.compile already applied, skipping")
            return

        logger.info("Applying torch.compile to transformers (post-LoRA)...")
        compile_start = time.perf_counter()

        # Enable TensorFloat-32 for additional speedup on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True

        # Regional compilation: only compile repeated WanTransformerBlock layers
        self.pipe.transformer.compile_repeated_blocks(
            mode="max-autotune",
            dynamic=True,
        )
        logger.info("Compiled transformer (high noise) - regional")

        self.pipe.transformer_2.compile_repeated_blocks(
            mode="max-autotune",
            dynamic=True,
        )
        logger.info("Compiled transformer_2 (low noise) - regional")

        compile_elapsed = time.perf_counter() - compile_start
        # Actual compilation is lazy and happens on first forward through compiled regions.
        self.timings["setup_compile_transformers"] = compile_elapsed
        logger.info(f"[TIMING] setup_compile_transformers: {compile_elapsed:.3f}s")

        self._compiled = True

    def _ensure_lora_hotswap_enabled(self, lora_paths: list):
        """
        For compiled + hotswap workflow: call enable_lora_hotswap() BEFORE the first load_lora_weights().
        """
        if getattr(self, "_lora_hotswap_enabled", False):
            return
        if self.CONFIG.lora_merge_method == LoraMergeMethod.FUSE:
            return

        target_rank = 32
        try:
            self.pipe.enable_lora_hotswap(target_rank=target_rank)
            self._lora_hotswap_enabled = True
            self._lora_hotswap_target_rank = target_rank
            logger.info(f"enable_lora_hotswap enabled with target_rank={target_rank}")
        except Exception as e:
            logger.warning(f"enable_lora_hotswap failed (proceeding without it): {e}")

    def _load_lora(self, lora_paths: list, lora_scale: Union[float, Dict[str, float]] = 1.0):
        """
        Load LoRA for Wan 2.2 I2V MoE model.

        Format: [{"low": "path", "high": "path"}] - supports single-side loading

        Supports two modes:
        - SET_ADAPTERS: Uses set_adapters for dynamic control (with hotswap for switching)
        - FUSE: Fuses weights for faster inference (requires reload for switching)

        Args:
            lora_paths: List with one MoE config dict
            lora_scale: LoRA weight scale (float or {"low": x, "high": y})
        """
        if not lora_paths or not isinstance(lora_paths[0], dict):
            raise ValueError("Wan 2.2 I2V requires MoE LoRA format: [{'low': 'path', 'high': 'path'}]")

        step_start = time.perf_counter()
        self._ensure_lora_hotswap_enabled(lora_paths)
        self.timings["enable_lora_hotswap"] = time.perf_counter() - step_start

        lora_config = lora_paths[0]
        high_path = lora_config.get("high")
        low_path = lora_config.get("low")

        adapter_names = []

        # Load high_noise LoRA -> transformer (default)
        if high_path and os.path.exists(high_path):
            high_dir = os.path.dirname(high_path)
            high_file = os.path.basename(high_path)
            logger.info(f"Loading high_noise LoRA: {high_file}")
            step_start = time.perf_counter()
            self.pipe.load_lora_weights(
                high_dir,
                weight_name=high_file,
                adapter_name="high_noise",
                local_files_only=True,
            )
            self.timings["load_high_lora"] = time.perf_counter() - step_start
            adapter_names.append("high_noise")
        elif high_path:
            logger.warning(f"high_noise LoRA not found: {high_path}")

        # Load low_noise LoRA -> transformer_2
        if low_path and os.path.exists(low_path):
            low_dir = os.path.dirname(low_path)
            low_file = os.path.basename(low_path)
            logger.info(f"Loading low_noise LoRA: {low_file} (load_into_transformer_2=True)")
            step_start = time.perf_counter()
            self.pipe.load_lora_weights(
                low_dir,
                weight_name=low_file,
                adapter_name="low_noise",
                local_files_only=True,
                load_into_transformer_2=True,  # Key: load to transformer_2
            )
            self.timings["load_low_lora"] = time.perf_counter() - step_start
            adapter_names.append("low_noise")
        elif low_path:
            logger.warning(f"low_noise LoRA not found: {low_path}")

        if not adapter_names:
            logger.warning("No LoRA loaded, using base model for inference")
            return

        self._adapter_names = adapter_names
        self._current_lora_paths = lora_paths

        if self.CONFIG.lora_merge_method == LoraMergeMethod.FUSE:
            # Fuse mode: merge LoRA weights into model for faster inference
            high_scale = lora_scale.get("high", 1.0) if isinstance(lora_scale, dict) else lora_scale
            low_scale = lora_scale.get("low", 1.0) if isinstance(lora_scale, dict) else lora_scale
            logger.info(f"Fusing dual LoRA with scale={{'high': {high_scale}, 'low': {low_scale}}}")
            if "high_noise" in adapter_names:
                step_start = time.perf_counter()
                self.pipe.fuse_lora(
                    adapter_names=["high_noise"],
                    lora_scale=high_scale,
                    components=["transformer"],
                )
                self.timings["fuse_high_lora"] = time.perf_counter() - step_start
            if "low_noise" in adapter_names:
                step_start = time.perf_counter()
                self.pipe.fuse_lora(
                    adapter_names=["low_noise"],
                    lora_scale=low_scale,
                    components=["transformer_2"],
                )
                self.timings["fuse_low_lora"] = time.perf_counter() - step_start
            # Unload adapter weights after fusing (optional, saves memory)
            step_start = time.perf_counter()
            self.pipe.unload_lora_weights()
            self.timings["unload_lora_weights"] = time.perf_counter() - step_start
            self._lora_fused = True
            self._num_loras_fused = len(adapter_names)
            logger.info(f"MoE LoRA fused: {adapter_names} with scale={{'high': {high_scale}, 'low': {low_scale}}}")
        else:
            # SET_ADAPTERS mode: use set_adapters for dynamic control
            if isinstance(lora_scale, dict):
                adapter_weights = [
                    lora_scale.get("high", 1.0) if name == "high_noise" else lora_scale.get("low", 1.0)
                    for name in adapter_names
                ]
            else:
                adapter_weights = [lora_scale] * len(adapter_names)
            step_start = time.perf_counter()
            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            self.timings["set_adapters"] = time.perf_counter() - step_start
            self._lora_fused = False
            logger.info(f"MoE LoRA loaded (adapter mode): {adapter_names} with scale={lora_scale}")

        self.lora_loaded = True
        self._current_lora_scale = lora_scale

        # Apply torch.compile AFTER LoRA loading for compatibility with adapters
        self._apply_torch_compile()

    def set_lora_scale(self, scale: Union[float, Dict[str, float]]) -> bool:
        """
        Set MoE LoRA scale (float applies to both; dict supports per-adapter scale).

        Note: In fuse mode, scale cannot be changed dynamically.
        Would require full model reload to apply new scale.

        Args:
            scale: New LoRA scale value or {"low": x, "high": y}

        Returns:
            True if successfully set, False if LoRA not loaded or fuse mode
        """
        if not self.lora_loaded:
            logger.debug("LoRA not loaded, skipping set_lora_scale")
            return False

        if scale == self._current_lora_scale:
            return True  # No change needed

        # Fuse mode: cannot change scale dynamically
        if self._lora_fused:
            logger.warning("Cannot change LoRA scale in fuse mode (dual LoRA)")
            return False

        try:
            if isinstance(scale, dict):
                adapter_weights = [
                    scale.get("high", 1.0) if name == "high_noise" else scale.get("low", 1.0)
                    for name in self._adapter_names
                ]
            else:
                # Adapter mode: apply same scale to all adapters
                adapter_weights = [scale] * len(self._adapter_names)
            self.pipe.set_adapters(self._adapter_names, adapter_weights=adapter_weights)
            self._current_lora_scale = scale
            logger.info(f"MoE LoRA scale set to {scale}")
            return True
        except Exception as e:
            logger.warning(f"Failed to set MoE lora_scale: {e}")
            return False

    def switch_lora(self, lora_paths: list, lora_scale: Union[float, Dict[str, float]] = 1.0) -> bool:
        """
        Switch dual LoRA for Wan22 MoE model.

        Two modes:
        - Adapter mode: Uses hotswap for in-place replacement (~1s)
        - Fuse mode: Requires full model reload (returns False)

        Args:
            lora_paths: New LoRA paths in MoE format [{"high": path, "low": path}]
            lora_scale: LoRA strength (float or {"low": x, "high": y})

        Returns:
            True if successfully switched, False if requires full reload
        """
        if not lora_paths or not isinstance(lora_paths[0], dict):
            logger.error("Wan22 requires MoE LoRA format: [{'low': 'path', 'high': 'path'}]")
            return False

        # Fuse mode: cannot switch without full reload (dual LoRA unfuse is unreliable)
        if self.CONFIG.lora_merge_method == LoraMergeMethod.FUSE and self._lora_fused:
            logger.info("Fuse mode with dual LoRA: requires full model reload for LoRA switch")
            return False

        lora_config = lora_paths[0]
        high_path = lora_config.get("high")
        low_path = lora_config.get("low")

        if not self.lora_loaded:
            # First load - use standard load
            logger.info("No LoRA currently loaded, using standard load")
            self._load_lora(lora_paths, lora_scale)
            return True

        # Adapter mode - use hotswap for each adapter
        logger.info("Switching dual LoRA using hotswap")
        try:
            # Hotswap high_noise adapter
            if high_path and os.path.exists(high_path):
                high_dir = os.path.dirname(high_path)
                high_file = os.path.basename(high_path)
                logger.info(f"Hotswapping high_noise LoRA: {high_file}")
                self.pipe.load_lora_weights(
                    high_dir,
                    weight_name=high_file,
                    adapter_name="high_noise",
                    local_files_only=True,
                    hotswap=True,
                )

            # Hotswap low_noise adapter
            if low_path and os.path.exists(low_path):
                low_dir = os.path.dirname(low_path)
                low_file = os.path.basename(low_path)
                logger.info(f"Hotswapping low_noise LoRA: {low_file}")
                self.pipe.load_lora_weights(
                    low_dir,
                    weight_name=low_file,
                    adapter_name="low_noise",
                    local_files_only=True,
                    load_into_transformer_2=True,
                    hotswap=True,
                )

            # Update scale and paths
            if isinstance(lora_scale, dict):
                adapter_weights = [
                    lora_scale.get("high", 1.0) if name == "high_noise" else lora_scale.get("low", 1.0)
                    for name in self._adapter_names
                ]
            else:
                adapter_weights = [lora_scale] * len(self._adapter_names)
            self.pipe.set_adapters(self._adapter_names, adapter_weights=adapter_weights)
            self._current_lora_scale = lora_scale
            self._current_lora_paths = lora_paths
            self._lora_high_path = high_path
            self._lora_low_path = low_path

            logger.info(f"Dual LoRA hotswapped successfully with scale={lora_scale}")
            return True

        except Exception as e:
            logger.warning(f"Dual LoRA hotswap failed: {e}, falling back to unload+load")
            if getattr(self, "_compiled", False):
                logger.error("hotswap failed on compiled pipeline; require full pipeline reload/recompile")
                return False
            try:
                # Fallback: unload all and reload
                self.pipe.unload_lora_weights()
                self.lora_loaded = False
                self._load_lora(lora_paths, lora_scale)
                return True
            except Exception as e2:
                logger.error(f"Fallback load also failed: {e2}")
            return False

    def _prepare_first_frame_conditioning(
        self,
        control_image: Image.Image,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Prepare first frame conditioning latent.

        Aligned with wan22_i2v_inference.py prepare_first_frame_conditioning function.
        """
        from toolkit.models.wan21.wan_utils import add_first_frame_conditioning

        device = self.device

        # Resize control image
        control_image = control_image.resize((width, height), Image.LANCZOS)

        # Prepare latents
        num_channels_latents = 16  # I2V uses 16 channels
        latents = self.pipe.prepare_latents(
            1,  # batch_size
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            torch.device(device),
            generator,
            None,
        ).to(self.dtype)

        # Convert first frame to [-1, 1] range
        first_frame_n1p1 = TF.to_tensor(control_image).unsqueeze(0).to(device, dtype=self.dtype) * 2.0 - 1.0

        # Use AI Toolkit's add_first_frame_conditioning function
        conditioned_latent = add_first_frame_conditioning(
            latent_model_input=latents,
            first_frame=first_frame_n1p1,
            vae=self.pipe.vae,
        )

        return conditioned_latent

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
        num_frames: int = 41,
        fps: int = 16,
    ) -> Dict[str, Any]:
        """Run Wan 2.2 I2V inference."""
        if control_image is None:
            raise ValueError("Wan 2.2 I2V requires a control image")

        # Prepare first frame conditioning
        conditioned_latent = self._prepare_first_frame_conditioning(
            control_image=control_image,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
        )

        extra_kwargs = {}
        # Fix for torch.compile + cudagraph_trees "overwritten by a subsequent run"
        # when multiple compiled regions are invoked within one denoising iteration.
        if getattr(self, "_compiled", False) and torch.cuda.is_available():
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                import inspect

                # Mark iteration begin for step 0 (before the first denoising step runs)
                torch.compiler.cudagraph_mark_step_begin()

                def _cg_step_end_callback(pipe, step_index, timestep, callback_kwargs):
                    # Mark iteration begin for the *next* step
                    torch.compiler.cudagraph_mark_step_begin()
                    return callback_kwargs

                sig = inspect.signature(self.pipe.__call__)
                if "callback_on_step_end" in sig.parameters:
                    extra_kwargs["callback_on_step_end"] = _cg_step_end_callback
                if "callback_on_step_end_tensor_inputs" in sig.parameters:
                    # We don't need tensors, just the hook
                    extra_kwargs["callback_on_step_end_tensor_inputs"] = []

        # Run inference
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=conditioned_latent,  # With first frame conditioning
            generator=generator,
            output_type="pil",
            **extra_kwargs,
        )

        frames = result.frames[0] if hasattr(result, "frames") else result.images

        return {"frames": frames, "fps": fps}

    def unload(self):
        """Unload the pipeline to free memory."""
        super().unload()
        self._lora_high_path = None
        self._lora_low_path = None
