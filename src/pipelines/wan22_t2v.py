"""
Wan 2.2 T2V 14B pipeline implementation.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Union

import torch
from PIL import Image
from diffusers import WanPipeline, WanTransformer3DModel
from accelerate.utils import compile_regions

from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from .torch_inductor_config import apply_torch_inductor_optimizations
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)

apply_torch_inductor_optimizations()


class Wan22T2V14BPipeline(BasePipeline):
    """
    Wan 2.2 T2V 14B pipeline.

    1. Load full pipeline from official model
    2. Replace transformer weights from ai-toolkit version
    3. Load dual LoRA (high_noise -> transformer, low_noise -> transformer_2)
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.WAN22_14B_T2V,
        base_model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        transformer_model="ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        is_video_model=True,
        default_num_frames=41,
        default_fps=16,
        enable_cpu_offload=False,
        enable_xformers=True,
    )

    def _load_pipeline(self):
        """
        Load Wan 2.2 T2V 14B pipeline with mixed model loading.

        Strategy
        1. Load full pipeline from official model (VAE, text_encoder, scheduler)
        2. Replace transformer weights from ai-toolkit version (aligned with training)
        """
        self.timings = {}
        total_start = time.perf_counter()
        logger.info(f"Loading pipeline: {self.CONFIG.base_model}")

        # 1. Load full pipeline from official model
        step_start = time.perf_counter()
        self.pipe = WanPipeline.from_pretrained(
            self.CONFIG.base_model,
            torch_dtype=self.dtype,
            token=self.hf_token,
            # We replace transformers from self.CONFIG.transformer_model below,
            # so skip downloading/loading them from the base model.
            transformer=None,
            transformer_2=None,
        )
        self.timings["load_base_pipeline"] = time.perf_counter() - step_start
        logger.info(f"[TIMING] load_base_pipeline: {self.timings['load_base_pipeline']:.3f}s")
        logger.info(f"Loaded pipeline. Scheduler: {type(self.pipe.scheduler).__name__}")

        # 2. Replace transformer weights from ai-toolkit version
        logger.info(f"Replacing transformer weights from: {self.CONFIG.transformer_model}")

        # Load transformer (high noise stage)
        step_start = time.perf_counter()
        transformer = WanTransformer3DModel.from_pretrained(
            self.CONFIG.transformer_model,
            subfolder="transformer",
            torch_dtype=self.dtype,
            token=self.hf_token,
        )
        self.timings["load_transformer"] = time.perf_counter() - step_start
        logger.info(f"[TIMING] load_transformer: {self.timings['load_transformer']:.3f}s")
        self.pipe.transformer = transformer
        logger.info("Replaced transformer (high noise)")

        # Load transformer_2 (low noise stage)
        step_start = time.perf_counter()
        transformer_2 = WanTransformer3DModel.from_pretrained(
            self.CONFIG.transformer_model,
            subfolder="transformer_2",
            torch_dtype=self.dtype,
            token=self.hf_token,
        )
        self.timings["load_transformer_2"] = time.perf_counter() - step_start
        logger.info(f"[TIMING] load_transformer_2: {self.timings['load_transformer_2']:.3f}s")
        self.pipe.transformer_2 = transformer_2
        logger.info("Replaced transformer_2 (low noise)")
        self.timings["load_total"] = time.perf_counter() - total_start
        logger.info(f"[TIMING] load_total: {self.timings['load_total']:.3f}s")

    def _enable_xformers(self):
        """
        Enable xformers memory efficient attention for both transformers.

        Override base method because WanPipeline uses dual transformer architecture
        and transformer_2 is manually added after pipeline creation.
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
        Load LoRA for Wan 2.2 14B MoE model.

        Format: [{"low": "path", "high": "path"}] - supports single-side loading

        Supports two modes:
        - SET_ADAPTERS: Uses set_adapters for dynamic control (with hotswap for switching)
        - FUSE: Fuses weights for faster inference (requires reload for switching)

        Args:
            lora_paths: List with one MoE config dict
            lora_scale: LoRA weight scale (float or {"low": x, "high": y})
        """
        if not lora_paths or not isinstance(lora_paths[0], dict):
            raise ValueError("Wan 2.2 14B requires MoE LoRA format: [{'low': 'path', 'high': 'path'}]")

        # enable_lora_hotswap must happen before the first load_lora_weights in compiled workflow
        step_start = time.perf_counter()
        self._ensure_lora_hotswap_enabled(lora_paths)
        self.timings["enable_lora_hotswap"] = time.perf_counter() - step_start

        lora_config = lora_paths[0]
        high_path = lora_config.get("high")
        low_path = lora_config.get("low")

        has_high = bool(high_path)
        has_low = bool(low_path)
        if not has_high and not has_low:
            raise ValueError("Wan 2.2 14B requires at least one LoRA path")

        fallback_path = high_path or low_path
        high_load_path = high_path or fallback_path
        low_load_path = low_path or fallback_path

        adapter_names = ["high_noise", "low_noise"]

        # Load high_noise LoRA -> transformer (default). Use placeholder if missing.
        if not has_high:
            logger.info("No high_noise LoRA provided; loading placeholder and setting scale=0")
        if high_load_path:
            high_dir = os.path.dirname(high_load_path)
            high_file = os.path.basename(high_load_path)
            logger.info(f"Loading high_noise LoRA: {high_file}")
            step_start = time.perf_counter()
            self.pipe.load_lora_weights(
                high_dir,
                weight_name=high_file,
                adapter_name="high_noise",
                local_files_only=True,
            )
            self.timings["load_high_lora"] = time.perf_counter() - step_start

        # Load low_noise LoRA -> transformer_2. Use placeholder if missing.
        if not has_low:
            logger.info("No low_noise LoRA provided; loading placeholder and setting scale=0")
        if low_load_path:
            low_dir = os.path.dirname(low_load_path)
            low_file = os.path.basename(low_load_path)
            logger.info(f"Loading low_noise LoRA: {low_file} (load_into_transformer_2=True)")
            step_start = time.perf_counter()
            self.pipe.load_lora_weights(
                low_dir,
                weight_name=low_file,
                adapter_name="low_noise",
                local_files_only=True,
                load_into_transformer_2=True,
            )
            self.timings["load_low_lora"] = time.perf_counter() - step_start

        self._adapter_names = adapter_names
        self._current_lora_paths = lora_paths

        if self.CONFIG.lora_merge_method == LoraMergeMethod.FUSE:
            if isinstance(lora_scale, dict):
                high_scale = lora_scale.get("high", 1.0)
                low_scale = lora_scale.get("low", 1.0)
            else:
                high_scale = lora_scale
                low_scale = lora_scale
            if not has_high:
                high_scale = 0.0
            if not has_low:
                low_scale = 0.0
            logger.info(f"Fusing dual LoRA with scale={{'high': {high_scale}, 'low': {low_scale}}}")
            # Wan LoRA fuse defaults to components=['transformer'].
            # For dual-denoiser Wan2.2, fuse transformer_2 explicitly when low_noise was loaded into transformer_2.
            if has_high and high_scale != 0.0:
                step_start = time.perf_counter()
                self.pipe.fuse_lora(
                    adapter_names=["high_noise"],
                    lora_scale=high_scale,
                    components=["transformer"],
                )
                self.timings["fuse_high_lora"] = time.perf_counter() - step_start
            if has_low and low_scale != 0.0:
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
            self._num_loras_fused = int(has_high and high_scale != 0.0) + int(has_low and low_scale != 0.0)
            logger.info(f"MoE LoRA fused: {adapter_names} with scale={{'high': {high_scale}, 'low': {low_scale}}}")
        else:
            # SET_ADAPTERS mode: use set_adapters for dynamic control
            if isinstance(lora_scale, dict):
                high_scale = lora_scale.get("high", 1.0)
                low_scale = lora_scale.get("low", 1.0)
            else:
                high_scale = lora_scale
                low_scale = lora_scale
            if not has_high:
                high_scale = 0.0
            if not has_low:
                low_scale = 0.0
            step_start = time.perf_counter()
            self.pipe.transformer.set_adapters(["high_noise"], [high_scale])
            self.pipe.transformer_2.set_adapters(["low_noise"], [low_scale])
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
            current_config = self._current_lora_paths[0] if self._current_lora_paths else {}
            has_high = bool(current_config.get("high"))
            has_low = bool(current_config.get("low"))

            if isinstance(scale, dict):
                high_scale = scale.get("high", 1.0)
                low_scale = scale.get("low", 1.0)
            else:
                high_scale = scale
                low_scale = scale
            if not has_high:
                high_scale = 0.0
            if not has_low:
                low_scale = 0.0

            self.pipe.transformer.set_adapters(["high_noise"], [high_scale])
            self.pipe.transformer_2.set_adapters(["low_noise"], [low_scale])
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
            if not high_path and not low_path:
                raise ValueError("No LoRA provided for switch_lora")

            # Hotswap high_noise adapter
            if high_path:
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
            if low_path:
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

            if isinstance(lora_scale, dict):
                high_scale = lora_scale.get("high", 1.0)
                low_scale = lora_scale.get("low", 1.0)
            else:
                high_scale = lora_scale
                low_scale = lora_scale
            if not high_path:
                high_scale = 0.0
            if not low_path:
                low_scale = 0.0

            self.pipe.transformer.set_adapters(["high_noise"], [high_scale])
            self.pipe.transformer_2.set_adapters(["low_noise"], [low_scale])
            self._adapter_names = ["high_noise", "low_noise"]
            self._current_lora_scale = lora_scale
            self._current_lora_paths = lora_paths

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
        """Run Wan 2.2 T2V inference."""
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

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            **extra_kwargs,
        )

        frames = result.frames[0] if hasattr(result, "frames") else result.images

        return {"frames": frames, "fps": fps}
