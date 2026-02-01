"""ComfyUI nodes for the full ai-toolkit-inference model catalog.

This module intentionally keeps imports lightweight; each node imports its pipeline
class lazily at execution time.

All nodes:
- return IMAGE (ComfyUI batch tensors)
- support optional LoRA path (single local path or URL) and lora_scale
  (Wan 2.2 14B MoE nodes instead use high/low path + scale inputs)
- support optional HuggingFace token

Video models return IMAGE batches where batch dimension is frames.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch

from .rc_common import (
    comfy_to_pil_image,
    get_or_load_pipeline,
    pil_frames_to_comfy_images,
    pil_to_comfy_image,
)

logger = logging.getLogger(__name__)


def _lora_path_to_list(lora_path: str) -> List[str]:
    """Convert a user-provided LoRA path field into the internal lora_paths list.

    Note: ComfyUI nodes currently support **one LoRA only** (either a local path or a URL).
    For Wan 2.2 14B MoE models we use dedicated high/low path + scale inputs.
    """
    if not lora_path or not lora_path.strip():
        return []
    s = lora_path.strip()
    if "," in s:
        raise ValueError(
            "Multiple LoRA stacking is not supported in these ComfyUI nodes yet. "
            "Please provide a single LoRA local path or URL in lora_path (no commas)."
        )
    return [s]


def _build_control_images(*imgs) -> List:
    out = []
    for t in imgs:
        if t is None:
            continue
        out.append(comfy_to_pil_image(t))
    return out


class _RCAitkBase:
    # override in subclasses
    MODEL_ID: str = ""
    DISPLAY_NAME: str = ""

    # defaults
    DEFAULT_WIDTH: int = 1024
    DEFAULT_HEIGHT: int = 1024
    DEFAULT_STEPS: int = 25
    DEFAULT_GUIDANCE: float = 4.0
    DEFAULT_SEED: int = 42

    # capabilities
    SUPPORTS_NEGATIVE: bool = True
    REQUIRES_CONTROL_IMAGE: bool = False
    # Number of control image slots exposed in the ComfyUI node UI.
    #
    # IMPORTANT:
    # - Pure T2I / T2V models should not expose a control image input.
    # - Edit / I2V models should set CONTROL_IMAGE_SLOTS >= 1 (and set
    #   REQUIRES_CONTROL_IMAGE=True if the model needs it).
    CONTROL_IMAGE_SLOTS: int = 0
    IS_VIDEO: bool = False
    DEFAULT_NUM_FRAMES: int = 41
    DEFAULT_FPS: int = 16

    # pipeline loading - default offload mode for this model class
    DEFAULT_OFFLOAD_MODE: str = "none"

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "prompt": ("STRING", {"multiline": True, "default": "a beautiful landscape"}),
            "width": ("INT", {"default": cls.DEFAULT_WIDTH, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": cls.DEFAULT_HEIGHT, "min": 64, "max": 4096, "step": 16}),
            "sample_steps": ("INT", {"default": cls.DEFAULT_STEPS, "min": 1, "max": 150}),
            "guidance_scale": ("FLOAT", {"default": float(cls.DEFAULT_GUIDANCE), "min": 0.0, "max": 20.0, "step": 0.1}),
            "seed": ("INT", {"default": cls.DEFAULT_SEED, "min": -1, "max": 0x7FFFFFFF}),
        }

        opt = {
            "offload_mode": (
                ["none", "model", "sequential"],
                {"default": cls.DEFAULT_OFFLOAD_MODE, "tooltip": "CPU offload strategy: none=full VRAM, model=lower VRAM, sequential=lowest VRAM (slowest)"},
            ),
            "lora_path": ("STRING", {"default": "", "tooltip": "LoRA local path or URL (single LoRA only)"}),
            "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "hf_token": ("STRING", {"default": ""}),
        }

        if cls.SUPPORTS_NEGATIVE:
            opt["negative_prompt"] = ("STRING", {"multiline": True, "default": ""})

        # control images
        if cls.CONTROL_IMAGE_SLOTS > 0:
            # primary control image
            if cls.REQUIRES_CONTROL_IMAGE:
                req["control_image"] = ("IMAGE",)
            else:
                opt["control_image"] = ("IMAGE",)

            # additional control images
            for i in range(2, cls.CONTROL_IMAGE_SLOTS + 1):
                opt[f"control_image_{i}"] = ("IMAGE",)

        # video controls
        if cls.IS_VIDEO:
            opt["num_frames"] = ("INT", {"default": cls.DEFAULT_NUM_FRAMES, "min": 1, "max": 201})
            opt["fps"] = ("INT", {"default": cls.DEFAULT_FPS, "min": 1, "max": 120})

        return {"required": req, "optional": opt}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "RunComfy-Inference"

    def _pipeline_ctor(self):
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        sample_steps: int,
        guidance_scale: float,
        seed: int,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        if prompt is None:
            raise ValueError("prompt is required")

        negative_prompt = kwargs.get("negative_prompt", "") if self.SUPPORTS_NEGATIVE else ""
        lora_scale_value: Union[float, Dict[str, float]] = float(kwargs.get("lora_scale", 1.0))
        hf_token = kwargs.get("hf_token", "") or None

        # Handle LoRA paths.
        lora_paths: List[Union[str, Dict[str, str]]] = _lora_path_to_list(kwargs.get("lora_path", ""))

        # Special handling for Wan22 14B MoE dual LoRA
        if self.MODEL_ID in ("wan22_14b_t2v", "wan22_14b_i2v"):
            high = kwargs.get("lora_path_high", "")
            low = kwargs.get("lora_path_low", "")
            d: Dict[str, str] = {}
            if high:
                d["high"] = high
            if low:
                d["low"] = low
            lora_paths = [d] if d else []

            has_scale_high = "lora_scale_high" in kwargs
            has_scale_low = "lora_scale_low" in kwargs
            if not has_scale_high and not has_scale_low:
                # Backward-compatible fallback: apply legacy single scale to both sides.
                legacy_scale = kwargs.get("lora_scale", 1.0)
                lora_scale_value = {"high": float(legacy_scale), "low": float(legacy_scale)}
            else:
                lora_scale_value = {
                    "high": float(kwargs.get("lora_scale_high", 1.0)),
                    "low": float(kwargs.get("lora_scale_low", 1.0)),
                }

        # Control images
        control_image = None
        control_images = None
        if self.CONTROL_IMAGE_SLOTS > 0:
            if kwargs.get("control_image") is not None:
                control_image = comfy_to_pil_image(kwargs["control_image"])

            # gather list for multi-control models
            extras = []
            for i in range(2, self.CONTROL_IMAGE_SLOTS + 1):
                k = f"control_image_{i}"
                if kwargs.get(k) is not None:
                    extras.append(comfy_to_pil_image(kwargs[k]))

            if extras:
                base = [control_image] if control_image is not None else []
                control_images = base + extras

        num_frames = int(kwargs.get("num_frames", self.DEFAULT_NUM_FRAMES)) if self.IS_VIDEO else None
        fps = int(kwargs.get("fps", self.DEFAULT_FPS)) if self.IS_VIDEO else None

        # Get offload_mode from input or use class default
        offload_mode = kwargs.get("offload_mode", self.DEFAULT_OFFLOAD_MODE)

        pipe = get_or_load_pipeline(
            model_id=self.MODEL_ID,
            pipeline_ctor=self._pipeline_ctor(),
            offload_mode=offload_mode,
            hf_token=hf_token,
            lora_paths=lora_paths,
            lora_scale=lora_scale_value,
        )

        # Special-case FLUX.2: the underlying ai-toolkit pipeline expects prompt-like
        # inputs to be non-None and list-like (it may call len() directly).
        # We bypass BasePipeline.generate() and call the vendor pipeline with safe
        # 1-item lists to avoid NoneType len() crashes and batch-size mis-inference.
        if self.MODEL_ID == "flux2":
            if negative_prompt:
                logger.warning("FLUX.2 does not support negative prompts, ignoring")

            # Match BasePipeline.generate seed handling.
            if seed < 0:
                seed = torch.randint(0, 2147483647, (1,)).item()
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            generator = torch.manual_seed(seed)

            # Build control_img_list like the Flux2Pipeline wrapper does.
            control_img_list = []
            if control_image is not None:
                if control_image.mode != "RGB":
                    control_image = control_image.convert("RGB")
                control_img_list.append(control_image)
            if control_images:
                for img in control_images:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    control_img_list.append(img)

            vendor = getattr(pipe, "pipe", None)
            if vendor is None or not callable(vendor):
                raise ValueError("FLUX.2 vendor pipeline is not loaded (pipe.pipe is missing)")

            # Prefer list-wrapped prompts (ai-toolkit FLUX.2 pipeline uses len()).
            try:
                vendor_result = vendor(
                    prompt=[str(prompt)],
                    negative_prompt=[""],
                    width=width,
                    height=height,
                    num_inference_steps=sample_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    control_img_list=control_img_list if control_img_list else None,
                )
            except TypeError:
                # Fallback if vendor unexpectedly wants raw strings.
                vendor_result = vendor(
                    prompt=str(prompt),
                    negative_prompt=" ",  # avoid len("") == 0 in buggy encode_prompt implementations
                    width=width,
                    height=height,
                    num_inference_steps=sample_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    control_img_list=control_img_list if control_img_list else None,
                )

            img0 = getattr(vendor_result, "images", [None])[0]
            if img0 is None:
                raise ValueError("FLUX.2 vendor pipeline returned no images")
            return (pil_to_comfy_image(img0),)

        result = pipe.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=sample_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            control_image=control_image,
            control_images=control_images,
            num_frames=num_frames,
            fps=fps,
        )

        if "image" in result:
            return (pil_to_comfy_image(result["image"]),)

        frames = result.get("frames")
        if frames:
            return (pil_frames_to_comfy_images(frames),)

        raise ValueError(f"Unexpected pipeline result keys: {list(result.keys())}")


# ===== Model-specific nodes =====

class RCZimage(_RCAitkBase):
    MODEL_ID = "zimage"
    DISPLAY_NAME = "RC Z-Image"
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 4.0

    def _pipeline_ctor(self):
        from src.pipelines.zimage import ZImagePipeline
        return ZImagePipeline


class RCZimageTurbo(_RCAitkBase):
    MODEL_ID = "zimage_turbo"
    DISPLAY_NAME = "RC Z-Image Turbo"
    DEFAULT_STEPS = 8
    DEFAULT_GUIDANCE = 1.0

    def _pipeline_ctor(self):
        from src.pipelines.zimage_turbo import ZImageTurboPipeline
        return ZImageTurboPipeline


class RCZimageDeturbo(_RCAitkBase):
    MODEL_ID = "zimage_deturbo"
    DISPLAY_NAME = "RC Z-Image De-Turbo"
    DEFAULT_STEPS = 25
    DEFAULT_GUIDANCE = 3.0

    def _pipeline_ctor(self):
        from src.pipelines.zimage_deturbo import ZImageDeturboPipeline
        return ZImageDeturboPipeline


class RCFluxDev(_RCAitkBase):
    MODEL_ID = "flux"
    DISPLAY_NAME = "RC FLUX.1-dev"
    DEFAULT_STEPS = 25
    DEFAULT_GUIDANCE = 4.0

    def _pipeline_ctor(self):
        from src.pipelines.flux_dev import FluxDevPipeline
        return FluxDevPipeline


class RCFluxKontext(_RCAitkBase):
    MODEL_ID = "flux_kontext"
    DISPLAY_NAME = "RC FLUX Kontext"
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 1

    def _pipeline_ctor(self):
        from src.pipelines.flux_kontext import FluxKontextPipeline
        return FluxKontextPipeline


class RCFlux2(_RCAitkBase):
    MODEL_ID = "flux2"
    DISPLAY_NAME = "RC FLUX.2"
    DEFAULT_STEPS = 25
    DEFAULT_GUIDANCE = 4.0

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        # FLUX.2 does not support negative prompts, but we keep the input for
        # workflow compatibility and to avoid surprising missing fields in older graphs.
        # The node will ignore it during execution.
        base["optional"]["negative_prompt"] = (
            "STRING",
            {
                "multiline": True,
                "default": "",
                "tooltip": "Ignored by FLUX.2 (kept for workflow compatibility).",
            },
        )
        return base

    def _pipeline_ctor(self):
        from src.pipelines.flux2 import Flux2Pipeline
        return Flux2Pipeline


class RCFlux2Klein4B(_RCAitkBase):
    MODEL_ID = "flux2_klein_4b"
    DISPLAY_NAME = "RC FLUX.2-klein 4B"
    DEFAULT_STEPS = 25
    DEFAULT_GUIDANCE = 4.0

    def _pipeline_ctor(self):
        from src.pipelines.flux2_klein import Flux2Klein4BPipeline
        return Flux2Klein4BPipeline


class RCFlux2Klein9B(_RCAitkBase):
    MODEL_ID = "flux2_klein_9b"
    DISPLAY_NAME = "RC FLUX.2-klein 9B"
    DEFAULT_STEPS = 25
    DEFAULT_GUIDANCE = 4.0

    def _pipeline_ctor(self):
        from src.pipelines.flux2_klein import Flux2Klein9BPipeline
        return Flux2Klein9BPipeline


class RCFlex1(_RCAitkBase):
    MODEL_ID = "flex1"
    DISPLAY_NAME = "RC Flex.1-alpha"

    def _pipeline_ctor(self):
        from src.pipelines.flex1_alpha import Flex1AlphaPipeline
        return Flex1AlphaPipeline


class RCFlex2(_RCAitkBase):
    MODEL_ID = "flex2"
    DISPLAY_NAME = "RC Flex.2"

    def _pipeline_ctor(self):
        from src.pipelines.flex2 import Flex2Pipeline
        return Flex2Pipeline


class RCSD15(_RCAitkBase):
    MODEL_ID = "sd15"
    DISPLAY_NAME = "RC SD 1.5"
    DEFAULT_GUIDANCE = 6.0

    def _pipeline_ctor(self):
        from src.pipelines.sd15 import SD15Pipeline
        return SD15Pipeline


class RCSDXL(_RCAitkBase):
    MODEL_ID = "sdxl"
    DISPLAY_NAME = "RC SDXL"
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 5.0

    def _pipeline_ctor(self):
        from src.pipelines.sdxl import SDXLPipeline
        return SDXLPipeline


class RCQwenImage(_RCAitkBase):
    MODEL_ID = "qwen_image"
    DISPLAY_NAME = "RC Qwen Image"

    def _pipeline_ctor(self):
        from src.pipelines.qwen_image import QwenImagePipeline
        return QwenImagePipeline


class RCQwenImage2512(_RCAitkBase):
    MODEL_ID = "qwen_image_2512"
    DISPLAY_NAME = "RC Qwen Image 2512"

    def _pipeline_ctor(self):
        from src.pipelines.qwen_image import QwenImage2512Pipeline
        return QwenImage2512Pipeline


class RCQwenImageEdit(_RCAitkBase):
    MODEL_ID = "qwen_image_edit"
    DISPLAY_NAME = "RC Qwen Image Edit"
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 1

    def _pipeline_ctor(self):
        from src.pipelines.qwen_image import QwenImageEditPipeline
        return QwenImageEditPipeline


class RCQwenImageEditPlus(_RCAitkBase):
    MODEL_ID = "qwen_image_edit_plus"
    DISPLAY_NAME = "RC Qwen Image Edit Plus"
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 3

    def _pipeline_ctor(self):
        from src.pipelines.qwen_image import QwenImageEditPlus2509Pipeline
        return QwenImageEditPlus2509Pipeline


class RCQwenImageEditPlus2511(_RCAitkBase):
    MODEL_ID = "qwen_image_edit_plus_2511"
    DISPLAY_NAME = "RC Qwen Image Edit Plus 2511"
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 3

    def _pipeline_ctor(self):
        from src.pipelines.qwen_image import QwenImageEditPlus2511Pipeline
        return QwenImageEditPlus2511Pipeline


class RCChroma(_RCAitkBase):
    MODEL_ID = "chroma"
    DISPLAY_NAME = "RC Chroma"

    def _pipeline_ctor(self):
        from src.pipelines.chroma import ChromaPipeline
        return ChromaPipeline


class RCHiDream(_RCAitkBase):
    MODEL_ID = "hidream"
    DISPLAY_NAME = "RC HiDream"
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 5.0

    def _pipeline_ctor(self):
        from src.pipelines.hidream import HiDreamPipeline
        return HiDreamPipeline


class RCHiDreamE1(_RCAitkBase):
    MODEL_ID = "hidream_e1"
    DISPLAY_NAME = "RC HiDream E1"
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 1
    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE = 5.0

    def _pipeline_ctor(self):
        from src.pipelines.hidream import HiDreamE1Pipeline
        return HiDreamE1Pipeline


class RCLumina2(_RCAitkBase):
    MODEL_ID = "lumina2"
    DISPLAY_NAME = "RC Lumina2"

    def _pipeline_ctor(self):
        from src.pipelines.lumina2 import Lumina2Pipeline
        return Lumina2Pipeline


class RCOmniGen2(_RCAitkBase):
    MODEL_ID = "omnigen2"
    DISPLAY_NAME = "RC OmniGen2"
    CONTROL_IMAGE_SLOTS = 5
    REQUIRES_CONTROL_IMAGE = False

    def _pipeline_ctor(self):
        from src.pipelines.omnigen2 import OmniGen2Pipeline
        return OmniGen2Pipeline


class RCLTX2(_RCAitkBase):
    MODEL_ID = "ltx2"
    DISPLAY_NAME = "RC LTX-2"
    IS_VIDEO = True
    DEFAULT_FPS = 24
    DEFAULT_OFFLOAD_MODE = "model"

    def _pipeline_ctor(self):
        from src.pipelines.ltx2 import LTX2Pipeline
        return LTX2Pipeline


# ===== Wan 2.1 =====

class RCWan21T2V14B(_RCAitkBase):
    MODEL_ID = "wan21_14b"
    DISPLAY_NAME = "RC Wan 2.1 T2V 14B"
    IS_VIDEO = True

    def _pipeline_ctor(self):
        from src.pipelines.wan21 import Wan21T2V14BPipeline
        return Wan21T2V14BPipeline


class RCWan21T2V1B(_RCAitkBase):
    MODEL_ID = "wan21_1b"
    DISPLAY_NAME = "RC Wan 2.1 T2V 1B"
    IS_VIDEO = True

    def _pipeline_ctor(self):
        from src.pipelines.wan21 import Wan21T2V1BPipeline
        return Wan21T2V1BPipeline


class RCWan21I2V14B(_RCAitkBase):
    MODEL_ID = "wan21_i2v_14b"
    DISPLAY_NAME = "RC Wan 2.1 I2V 14B"
    IS_VIDEO = True
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 1

    def _pipeline_ctor(self):
        from src.pipelines.wan21 import Wan21I2V14BPipeline
        return Wan21I2V14BPipeline


class RCWan21I2V14B480P(_RCAitkBase):
    MODEL_ID = "wan21_i2v_14b480p"
    DISPLAY_NAME = "RC Wan 2.1 I2V 14B 480p"
    IS_VIDEO = True
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 1

    def _pipeline_ctor(self):
        from src.pipelines.wan21 import Wan21I2V14B480PPipeline
        return Wan21I2V14B480PPipeline


# ===== Wan 2.2 =====

class RCWan22T2V14B(_RCAitkBase):
    MODEL_ID = "wan22_14b_t2v"
    DISPLAY_NAME = "RC Wan 2.2 T2V 14B"
    IS_VIDEO = True
    DEFAULT_OFFLOAD_MODE = "model"

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"].pop("lora_path", None)
        base["optional"].pop("lora_scale", None)
        base["optional"]["lora_path_high"] = ("STRING", {"default": ""})
        base["optional"]["lora_path_low"] = ("STRING", {"default": ""})
        base["optional"]["lora_scale_high"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
        base["optional"]["lora_scale_low"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
        return base

    def _pipeline_ctor(self):
        from src.pipelines.wan22_t2v import Wan22T2V14BPipeline
        return Wan22T2V14BPipeline


class RCWan22I2V14B(_RCAitkBase):
    MODEL_ID = "wan22_14b_i2v"
    DISPLAY_NAME = "RC Wan 2.2 I2V 14B"
    IS_VIDEO = True
    REQUIRES_CONTROL_IMAGE = True
    CONTROL_IMAGE_SLOTS = 1

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"].pop("lora_path", None)
        base["optional"].pop("lora_scale", None)
        base["optional"]["lora_path_high"] = ("STRING", {"default": ""})
        base["optional"]["lora_path_low"] = ("STRING", {"default": ""})
        base["optional"]["lora_scale_high"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
        base["optional"]["lora_scale_low"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
        return base

    def _pipeline_ctor(self):
        from src.pipelines.wan22_i2v import Wan22I2V14BPipeline
        return Wan22I2V14BPipeline


class RCWan22TI2V5B(_RCAitkBase):
    MODEL_ID = "wan22_5b"
    DISPLAY_NAME = "RC Wan 2.2 TI2V 5B"
    IS_VIDEO = True

    def _pipeline_ctor(self):
        from src.pipelines.wan22_5b import Wan22TI2V5BPipeline
        return Wan22TI2V5BPipeline
