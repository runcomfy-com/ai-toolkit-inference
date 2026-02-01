"""
Base pipeline class with configuration.
"""

import os
import gc
import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

import torch
from PIL import Image

from ..schemas.models import ModelType

# Lazy imports - diffusers is only imported when actually needed
# This avoids loading all diffusers pipelines at startup

logger = logging.getLogger(__name__)


# CPU offload mode: exactly one strategy is applied during pipeline load.
# - "none": No offload, model stays on GPU (fastest inference, highest VRAM)
# - "model": Model CPU offload - moves full model between CPU/GPU (balanced)
# - "sequential": Sequential CPU offload - moves layers one at a time (lowest VRAM, slowest)
OffloadMode = Literal["none", "model", "sequential"]

# Progress/interrupt observer hook for integrations (e.g. ComfyUI).
#
# Design goal:
# - Keep pipelines framework-agnostic: the pipeline code never imports ComfyUI.
# - External callers (ComfyUI nodes, server, etc.) may install an observer for the
#   duration of a generate() call to receive per-step notifications and/or raise
#   an exception to interrupt generation.
#
# Observer signature: (step_index, total_steps, timestep) -> None
PipelineStepObserver = Callable[[int, int, Any], None]
_PIPELINE_STEP_OBSERVER: ContextVar[Optional[PipelineStepObserver]] = ContextVar(
    "aitk_pipeline_step_observer",
    default=None,
)


@contextmanager
def pipeline_step_observer(observer: Optional[PipelineStepObserver]) -> Iterator[None]:
    """Temporarily install a per-step observer for diffusers callbacks."""
    token = _PIPELINE_STEP_OBSERVER.set(observer)
    try:
        yield
    finally:
        _PIPELINE_STEP_OBSERVER.reset(token)


class LoraMergeMethod(Enum):
    """
    LoRA merge method determines how LoRA weights are applied to the base model.

    SET_ADAPTERS: Uses PEFT set_adapters() for dynamic control.
        - Switch strategy: hotswap (preferred) > unload+load (fallback)
        - Supports dynamic lora_scale change
        - Fastest switching (~0.16s with hotswap)

    FUSE: Uses fuse_lora() to merge weights into base model.
        - Switch strategy: unfuse -> load -> fuse
        - Scale set during fuse_lora(), cannot change dynamically
        - Faster inference but slower switching (~0.35s)
        - Cannot reliably unfuse if multiple LoRAs were fused

    CUSTOM: Pipeline handles LoRA loading/switching itself.
        - For models with special requirements (e.g., Flux2 manual merge, Chroma quantization)
        - switch_lora() returns False, requires full model reload
    """

    SET_ADAPTERS = "set_adapters"
    FUSE = "fuse"
    CUSTOM = "custom"


@dataclass
class PipelineConfig:
    """
    Configuration for a pipeline.

    Each pipeline subclass defines its config via class attributes.
    """

    model_type: ModelType
    base_model: str
    resolution_divisor: int
    default_steps: int
    default_guidance_scale: float
    requires_control_image: bool = False
    supports_negative_prompt: bool = True
    is_video_model: bool = False
    default_num_frames: int = 1
    default_fps: int = 16

    # LoRA merge method (replaces use_fuse_lora and supports_lora_adapter)
    lora_merge_method: LoraMergeMethod = LoraMergeMethod.SET_ADAPTERS

    # Reserved: support for multiple LoRAs simultaneously
    support_multiple_loras: bool = False

    # For models with separate transformer loading
    transformer_model: Optional[str] = None
    # CPU offload for large models (14B+)
    enable_cpu_offload: bool = False
    # Additional defaults
    default_width: int = 1024
    default_height: int = 1024
    default_neg: str = ""
    default_seed: int = 42
    default_network_multiplier: float = 1.0
    # Enable xformers memory efficient attention
    enable_xformers: bool = False


class BasePipeline(ABC):
    """
    Base class for all inference pipelines.

    Subclasses must:
    1. Define CONFIG as a class attribute with PipelineConfig
    2. Implement _load_pipeline
    3. Implement _run_inference
    """

    # Subclasses MUST override this
    CONFIG: PipelineConfig = None

    def __init__(
        self,
        device: str = "cuda",
        offload_mode: OffloadMode = "model",
        hf_token: Optional[str] = None,
    ):
        if self.CONFIG is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define CONFIG class attribute")

        self.device = device
        self.hf_token = hf_token
        self.offload_mode: OffloadMode = offload_mode

        self.pipe = None
        self.lora_loaded = False
        self.dtype = self._default_dtype_for_device(device)
        self.timings: Dict[str, float] = {}
        self._current_lora_scale: float = 1.0
        self._current_lora_paths: List[str] = []  # Track currently loaded LoRA paths
        self._lora_fused: bool = False  # Track if LoRA is fused into model weights
        self._num_loras_fused: int = 0  # Track number of LoRAs fused (for unfuse reliability check)

    @staticmethod
    def _default_dtype_for_device(device: str) -> torch.dtype:
        """Pick a safe default dtype based on hardware capabilities.

        Rules:
        - CPU: float32
        - CUDA: bf16 if supported, else float16
        - Other/unknown: float32
        """
        dev = (device or "").lower()
        if dev.startswith("cpu") or dev == "cpu":
            return torch.float32

        if dev.startswith("cuda") and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                # Older torch builds may not expose is_bf16_supported(); fall back to fp16.
                pass
            return torch.float16

        return torch.float32

    def _get_execution_device(self) -> torch.device:
        """Best-effort device selection for input tensors.

        Prefer diffusers' `_execution_device` (used for CPU offload modes), fall back
        to pipeline `.device`, then finally to this wrapper's configured device.
        """
        pipe_obj = getattr(self, "pipe", None)
        if pipe_obj is not None:
            # diffusers (and many wrappers) expose this for correct input placement under offload
            try:
                exec_dev = getattr(pipe_obj, "_execution_device", None)
                if exec_dev is not None:
                    return exec_dev if isinstance(exec_dev, torch.device) else torch.device(exec_dev)
            except Exception:
                pass

            # fallback: some pipelines expose `.device`
            try:
                dev = getattr(pipe_obj, "device", None)
                if dev is not None:
                    return dev if isinstance(dev, torch.device) else torch.device(dev)
            except Exception:
                pass

        # wrapper-level fallback
        dev_s = (getattr(self, "device", None) or "cpu")
        dev_s = str(dev_s)
        if dev_s.startswith("cuda") and torch.cuda.is_available():
            return torch.device(dev_s)
        if dev_s.startswith("mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def get_config(cls) -> PipelineConfig:
        """Get pipeline configuration."""
        return cls.CONFIG

    def load(self, lora_paths: List[str], lora_scale: float = 1.0):
        """Load the pipeline and LoRA weights."""
        logger.info(f"Loading pipeline: {self.CONFIG.base_model}")

        # Load the base pipeline
        self._load_pipeline()

        # Enable xformers memory efficient attention if available
        # This provides ~20% speedup for Transformer-based models (Wan, FLUX, etc.)
        self._enable_xformers()

        # Apply exactly ONE offload strategy based on offload_mode.
        # This is the single place where device placement is configured.
        self._apply_offload_mode()

        # Load LoRA weights
        if lora_paths:
            self._load_lora(lora_paths, lora_scale)
        else:
            logger.warning("No LoRA paths provided")

    def _apply_offload_mode(self):
        """Apply the configured offload mode to the pipeline.
        
        This method applies exactly ONE offload strategy:
        - "sequential": Sequential CPU offload (lowest VRAM, slowest)
        - "model": Model CPU offload (balanced)
        - "none": No offload, model on GPU (fastest, highest VRAM)
        
        Subclasses can override this if they need custom offload behavior.
        """
        if self.pipe is None:
            return

        if self.offload_mode == "sequential":
            self._enable_sequential_cpu_offload()
        elif self.offload_mode == "model":
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                logger.info("Enabling model CPU offload")
                self.pipe.enable_model_cpu_offload()
            else:
                logger.warning("Model CPU offload not available, moving to device")
                self.pipe.to(self.device)
        else:  # "none"
            logger.info(f"Moving pipeline to {self.device} (no offload)")
            self.pipe.to(self.device)

    def _enable_sequential_cpu_offload(self):
        """Enable sequential CPU offload for lowest VRAM usage.
        
        Sequential offload moves layers one at a time between CPU and GPU,
        which uses minimal VRAM but is slower than model offload.
        
        Subclasses can override this to add additional state tracking
        (e.g., QwenImagePipeline sets _using_sequential_offload).
        """
        if self.pipe is None:
            return

        if hasattr(self.pipe, "enable_sequential_cpu_offload"):
            logger.info("Enabling sequential CPU offload (low VRAM mode)")
            self.pipe.enable_sequential_cpu_offload()
        else:
            logger.warning("Sequential CPU offload not available, falling back to model offload")
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)

    def _inject_diffusers_callback_kwargs(
        self,
        call_kwargs: Dict[str, Any],
        *,
        total_steps: int,
        pipe: Optional[Any] = None,
    ) -> None:
        """Inject step callbacks into a diffusers pipeline call.

        This is intentionally integration-agnostic: it only consults the current
        observer installed via `pipeline_step_observer(...)`. If no observer is
        installed, this is a no-op.

        Preference order:
        - diffusers `callback_on_step_end` (newer API)
        - diffusers `callback` + `callback_steps` (legacy API)

        Args:
            call_kwargs: kwargs dict that will be passed to the pipeline call (mutated in-place).
            total_steps: total number of inference steps for progress reporting.
            pipe: optional pipeline object to inspect; defaults to `self.pipe`.
        """
        observer = _PIPELINE_STEP_OBSERVER.get()
        if observer is None:
            return

        pipe_obj = pipe if pipe is not None else self.pipe
        if pipe_obj is None:
            return

        pipe_call = getattr(pipe_obj, "__call__", None)
        if pipe_call is None:
            return

        try:
            sig = inspect.signature(pipe_call)
        except (TypeError, ValueError):
            return

        params = sig.parameters

        def _notify(step, timestep):
            try:
                step_i = int(step) if step is not None else 0
            except Exception:
                step_i = 0
            try:
                total_i = int(total_steps)
            except Exception:
                total_i = total_steps  # type: ignore[assignment]
            # Let observer raise to interrupt generation.
            observer(step_i, total_i, timestep)

        # Newer diffusers API: callback_on_step_end
        if "callback_on_step_end" in params:
            existing_cb = call_kwargs.get("callback_on_step_end", None)

            def callback_on_step_end(*args, **kwargs):
                # Typical signatures:
                # - (pipe, step, timestep, callback_kwargs)
                # - (step, timestep, callback_kwargs)
                step = kwargs.get("step", None)
                timestep = kwargs.get("timestep", None)
                cb_kwargs = None

                # Run any existing callback first (e.g., internal hooks), preserving its return value.
                if callable(existing_cb):
                    try:
                        cb_kwargs = existing_cb(*args, **kwargs)
                    except Exception:
                        # Let exceptions propagate (interrupts must stop generation).
                        raise

                # Fallback to diffusers-provided callback_kwargs if the existing callback didn't return one.
                if cb_kwargs is None:
                    cb_kwargs = kwargs.get("callback_kwargs", None)

                if step is None:
                    if len(args) >= 2 and isinstance(args[1], int):
                        step = args[1]
                        if timestep is None and len(args) >= 3:
                            timestep = args[2]
                    elif len(args) >= 1 and isinstance(args[0], int):
                        step = args[0]
                        if timestep is None and len(args) >= 2:
                            timestep = args[1]
                    else:
                        step = 0

                # diffusers typically passes callback_kwargs as the last positional arg
                if cb_kwargs is None and args and isinstance(args[-1], dict):
                    cb_kwargs = args[-1]
                if cb_kwargs is None:
                    cb_kwargs = {}

                _notify(step, timestep)
                return cb_kwargs

            call_kwargs["callback_on_step_end"] = callback_on_step_end
            if "callback_on_step_end_tensor_inputs" in params and "callback_on_step_end_tensor_inputs" not in call_kwargs:
                # We only need step/timestep for progress + interrupt, so no tensors required.
                call_kwargs["callback_on_step_end_tensor_inputs"] = []
            return

        # Legacy diffusers API: callback (+ callback_steps)
        if "callback" in params:
            existing_cb = call_kwargs.get("callback", None)

            def callback(*args, **kwargs):
                # Typical signature: (step, timestep, latents)
                if callable(existing_cb):
                    try:
                        existing_cb(*args, **kwargs)
                    except Exception:
                        raise
                step = kwargs.get("step", None)
                timestep = kwargs.get("timestep", None)
                if step is None:
                    step = args[0] if len(args) >= 1 else 0
                if timestep is None:
                    timestep = args[1] if len(args) >= 2 else None
                _notify(step, timestep)

            call_kwargs["callback"] = callback
            if "callback_steps" in params and "callback_steps" not in call_kwargs:
                call_kwargs["callback_steps"] = 1

    @abstractmethod
    def _load_pipeline(self):
        """Load the diffusers pipeline. Must set self.pipe."""
        pass

    def _enable_xformers(self):
        """
        Enable xformers memory efficient attention if available.

        Only enabled for pipelines with CONFIG.enable_xformers=True.
        Provides ~20% speedup for Transformer-based models.
        Falls back to PyTorch SDPA (default in PyTorch 2.0+) if xformers unavailable.
        """
        if self.pipe is None:
            return

        # Only enable if configured
        if not self.CONFIG.enable_xformers:
            logger.debug(f"xformers disabled for {self.__class__.__name__}")
            return

        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention enabled")
            except Exception as e:
                logger.debug(f"xformers not available, using default attention: {e}")
        else:
            logger.debug("Pipeline does not support xformers")

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0, hotswap: bool = False):
        """
        Load LoRA weights.

        Default implementation uses diffusers load_lora_weights.
        Override for custom behavior (e.g., dual LoRA for Wan22).

        Args:
            lora_paths: List of paths to LoRA weight files (or list with dict for MoE)
            lora_scale: LoRA strength (0.0 to 2.0)
            hotswap: If True, use hotswap mode (in-place replacement, requires existing LoRA loaded)
        """
        if not lora_paths:
            logger.warning("No LoRA paths provided")
            return

        # Default: load first LoRA only (for single-LoRA models)
        first_lora = lora_paths[0]

        # Handle MoE format (dict) - extract first path
        if isinstance(first_lora, dict):
            lora_path = list(first_lora.values())[0] if first_lora else None
            if not lora_path:
                logger.warning("Empty MoE config provided")
                return
        else:
            # Standard format (string path)
            lora_path = first_lora

        merge_method = self.CONFIG.lora_merge_method
        logger.info(f"Loading LoRA: {lora_path} (method={merge_method.value}, hotswap={hotswap})")

        lora_dir = os.path.dirname(lora_path)
        lora_file = os.path.basename(lora_path)

        # Load LoRA weights
        # hotswap=True does in-place replacement, only works when LoRA already loaded
        self.pipe.load_lora_weights(
            lora_dir,
            weight_name=lora_file,
            adapter_name="lora",
            local_files_only=True,
            hotswap=hotswap,
        )

        if merge_method == LoraMergeMethod.FUSE:
            # Fuse mode: merge LoRA weights into base model
            # Scale is set in fuse_lora, not dynamically changeable
            self.pipe.fuse_lora(adapter_names=["lora"], lora_scale=lora_scale)
            # Unload PEFT adapters to free memory after fusion
            self.pipe.unload_lora_weights()
            self._lora_fused = True
            self._num_loras_fused = 1  # Track for unfuse reliability
        else:
            # SET_ADAPTERS mode: keep as adapter for dynamic control
            # load_lora_weights already activates LoRA at scale=1.0
            self._lora_fused = False
            self._num_loras_fused = 0
            if lora_scale != 1.0:
                self._set_lora_scale_direct(lora_scale)

        self.lora_loaded = True
        self._current_lora_scale = lora_scale
        self._current_lora_paths = lora_paths  # Track current LoRA paths
        logger.info(f"LoRA loaded with scale={lora_scale}, method={merge_method.value}")

    def _set_lora_scale_direct(self, scale: float):
        """
        Set LoRA scale using low-level PEFT API.

        This bypasses pipe.set_adapters() which may not be supported by all models.
        Works by directly calling set_scale on all PEFT LoRA layers.

        Args:
            scale: LoRA scale value
        """
        # Lazy import to avoid loading all diffusers pipelines at startup
        from diffusers.utils.peft_utils import set_weights_and_activate_adapters

        # Get all components that may have LoRA adapters
        for component_name in ["transformer", "unet", "text_encoder", "text_encoder_2"]:
            component = getattr(self.pipe, component_name, None)
            if component is not None:
                try:
                    set_weights_and_activate_adapters(component, ["lora"], [scale])
                except Exception as e:
                    logger.debug(f"Could not set scale for {component_name}: {e}")

    def set_lora_scale(self, scale: float) -> bool:
        """
        Dynamically set LoRA scale for per-prompt control.

        Behavior depends on lora_merge_method:
        - SET_ADAPTERS: Direct scale change via PEFT API (fast)
        - FUSE: Requires unfuse -> reload -> fuse(new_scale) (slow)
        - CUSTOM: Not supported, returns False

        Args:
            scale: New LoRA scale value

        Returns:
            True if successfully set, False if not supported or LoRA not loaded
        """
        if not self.lora_loaded:
            logger.debug("LoRA not loaded, skipping set_lora_scale")
            return False

        if scale == self._current_lora_scale:
            return True  # No change needed

        merge_method = self.CONFIG.lora_merge_method

        if merge_method == LoraMergeMethod.CUSTOM:
            logger.warning("CUSTOM merge method does not support dynamic scale change")
            return False

        try:
            if merge_method == LoraMergeMethod.FUSE:
                # Fuse mode: need unfuse -> reload -> fuse(new_scale)
                # Only works for single fused LoRA
                if self._num_loras_fused > 1:
                    logger.warning("Cannot change scale for multiple fused LoRAs")
                    return False

                logger.info(f"Changing fused LoRA scale: unfuse -> reload -> fuse(scale={scale})")
                self.pipe.unfuse_lora()
                self._lora_fused = False

                # Reload LoRA weights (they were unloaded after fuse)
                first_lora = self._current_lora_paths[0]
                if isinstance(first_lora, dict):
                    lora_path = list(first_lora.values())[0]
                else:
                    lora_path = first_lora

                lora_dir = os.path.dirname(lora_path)
                lora_file = os.path.basename(lora_path)
                self.pipe.load_lora_weights(
                    lora_dir,
                    weight_name=lora_file,
                    adapter_name="lora",
                    local_files_only=True,
                )
                # Fuse with new scale
                self.pipe.fuse_lora(adapter_names=["lora"], lora_scale=scale)
                self.pipe.unload_lora_weights()
                self._lora_fused = True
            else:
                # SET_ADAPTERS mode: direct scale change via PEFT API
                self._set_lora_scale_direct(scale)

            self._current_lora_scale = scale
            logger.info(f"LoRA scale set to {scale}")
            return True
        except Exception as e:
            logger.warning(f"Failed to set lora_scale: {e}")
            return False

    def switch_lora(self, lora_paths: list, lora_scale: float = 1.0) -> bool:
        """
        Switch to a new LoRA without reloading the base model.

        Strategy depends on lora_merge_method:
        - SET_ADAPTERS: hotswap (preferred) > unload+load (fallback)
        - FUSE: unfuse -> load -> fuse
        - CUSTOM: returns False (requires full model reload)

        Args:
            lora_paths: New LoRA paths to load
            lora_scale: LoRA strength (0.0 to 2.0)

        Returns:
            True if successfully switched, False if not supported (requires full reload)
        """
        if self.pipe is None:
            logger.warning("Pipeline not loaded, cannot switch LoRA")
            return False

        merge_method = self.CONFIG.lora_merge_method

        if merge_method == LoraMergeMethod.CUSTOM:
            logger.info(f"Pipeline {self.__class__.__name__} uses CUSTOM merge, requires full reload")
            return False

        if merge_method == LoraMergeMethod.FUSE:
            # Fuse mode: unfuse -> load -> fuse
            return self._switch_lora_fuse_mode(lora_paths, lora_scale)
        else:
            # SET_ADAPTERS mode: hotswap > unload+load
            return self._switch_lora_set_adapters_mode(lora_paths, lora_scale)

    def _switch_lora_set_adapters_mode(self, lora_paths: list, lora_scale: float = 1.0) -> bool:
        """
        Switch LoRA in SET_ADAPTERS mode.

        Strategy: hotswap (preferred) > unload+load (fallback)

        Hotswap is faster but:
        - Only works when LoRA is already loaded
        - Does NOT support text encoder LoRA
        """
        if not self.lora_loaded:
            # First load - cannot use hotswap
            logger.info("No LoRA currently loaded, using standard load (hotswap=False)")
            self._load_lora(lora_paths, lora_scale, hotswap=False)
            return True

        # Try hotswap first (in-place replacement)
        logger.info("Switching LoRA using hotswap (in-place replacement)")
        try:
            self._load_lora(lora_paths, lora_scale, hotswap=True)
            return True
        except Exception as e:
            logger.warning(f"Hotswap failed: {e}, falling back to unload+load")
            # Fallback: unload and reload
            try:
                self.pipe.unload_lora_weights()
                self.lora_loaded = False
                self._current_lora_paths = []
                self._load_lora(lora_paths, lora_scale, hotswap=False)
                return True
            except Exception as e2:
                logger.error(f"Fallback load also failed: {e2}")
                return False

    def _switch_lora_fuse_mode(self, lora_paths: list, lora_scale: float = 1.0) -> bool:
        """
        Switch LoRA using unfuse -> load -> fuse.

        Note: unfuse_lora() only works reliably when single LoRA was fused.
        Multiple fused LoRAs require full model reload.
        """
        if self.lora_loaded and getattr(self, "_lora_fused", False):
            # Check if multiple LoRAs were fused - unfuse is unreliable in this case
            if self._num_loras_fused > 1:
                logger.warning(
                    f"Multiple LoRAs ({self._num_loras_fused}) were fused, "
                    "unfuse_lora() is unreliable. Requires full model reload."
                )
                return False  # Signal caller to reload model

            logger.info("Unfusing current LoRA weights")
            try:
                self.pipe.unfuse_lora()
                self._lora_fused = False
                self._num_loras_fused = 0
            except Exception as e:
                logger.warning(f"Failed to unfuse LoRA: {e}, may need full reload")
                return False

        # Load and fuse new LoRA
        logger.info("Loading and fusing new LoRA")
        try:
            self._load_lora(lora_paths, lora_scale, hotswap=False)
            return True
        except Exception as e:
            logger.error(f"Failed to load new LoRA: {e}")
            return False

    def get_current_lora_paths(self) -> List[str]:
        """Get the currently loaded LoRA paths."""
        return self._current_lora_paths

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 42,
        control_image: Optional[Image.Image] = None,
        control_images: Optional[List[Image.Image]] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        output_type: str = "pil",
        latents: Optional[torch.Tensor] = None,
        denoise_strength: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate image/video.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Output width
            height: Output height
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed (-1 for random)
            control_image: Single control image (for most models)
            control_images: Multiple control images (for qwen_image_edit_plus etc.)
            num_frames: Number of frames for video
            fps: Frames per second for video
            output_type: "pil" for PIL Image, "latent" for raw latent tensor
            latents: Optional input latents for img2img/refine workflows
            denoise_strength: Denoising strength (0-1) when using input latents

        Returns:
            Dict with "image" or "frames" or "latents" key, plus "seed".
        """
        # Apply defaults from config
        if num_inference_steps is None:
            num_inference_steps = self.CONFIG.default_steps
        if guidance_scale is None:
            guidance_scale = self.CONFIG.default_guidance_scale
        if num_frames is None:
            num_frames = self.CONFIG.default_num_frames
        if fps is None:
            fps = self.CONFIG.default_fps

        # Validate and snap dimensions to resolution_divisor (floor + warn)
        divisor = self.CONFIG.resolution_divisor
        snapped_width = (width // divisor) * divisor
        snapped_height = (height // divisor) * divisor
        if snapped_width != width or snapped_height != height:
            logger.warning(
                f"Resolution {width}x{height} not divisible by {divisor}; "
                f"snapping to {snapped_width}x{snapped_height}"
            )
            width = snapped_width
            height = snapped_height

        # Handle seed
        if seed < 0:
            seed = torch.randint(0, 2147483647, (1,)).item()

        # Stop global RNG side-effects: use a local generator (CPU generator for ai-toolkit compatibility)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        logger.info(f"Generating: prompt='{prompt[:50]}...', size={width}x{height}, seed={seed}")

        # Run inference - try with output_type/latents support, fall back to basic call
        with torch.inference_mode():
            try:
                result = self._run_inference(
                    prompt=prompt,
                    negative_prompt=negative_prompt or "",
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    control_image=control_image,
                    control_images=control_images,
                    num_frames=num_frames,
                    fps=fps,
                    output_type=output_type,
                    latents=latents,
                    denoise_strength=denoise_strength,
                )
            except TypeError:
                # Fallback for pipelines that don't support output_type/latents yet
                if output_type != "pil" or latents is not None:
                    raise NotImplementedError(
                        f"{self.__class__.__name__} does not support output_type='{output_type}' or latents input yet"
                    )
                result = self._run_inference(
                    prompt=prompt,
                    negative_prompt=negative_prompt or "",
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    control_image=control_image,
                    control_images=control_images,
                    num_frames=num_frames,
                    fps=fps,
                )

        # Add seed to result
        result["seed"] = seed

        return result

    @abstractmethod
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
        """
        Run inference.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Output width
            height: Output height
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            generator: Random generator
            control_image: Single control image (for most models)
            control_images: Multiple control images (for qwen_image_edit_plus etc.)
            num_frames: Number of frames for video
            fps: Frames per second for video

        Must return:
        - For image models: {"image": PIL.Image}
        - For video models: {"frames": List[PIL.Image]}
        """
        pass

    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        self.lora_loaded = False
        self._current_lora_paths = []
        self._lora_fused = False
        self._num_loras_fused = 0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Pipeline unloaded")
