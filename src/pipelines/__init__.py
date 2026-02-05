"""
Pipeline implementations for different models.

Each pipeline defines its own CONFIG (PipelineConfig) as a class attribute.

IMPORTANT: This module uses lazy imports to avoid loading all diffusers pipelines
at module load time. Pipeline classes are only imported when actually accessed.
This allows ComfyUI to load the node pack even if some diffusers dependencies
are missing (e.g., MT5Tokenizer for HunyuanDiT).
"""

from typing import Type, Optional, TYPE_CHECKING

# These are safe to import eagerly (no diffusers dependency)
from .base import BasePipeline, PipelineConfig, LoraMergeMethod
from ..schemas.models import ModelType

# For type checking only - not actually imported at runtime
if TYPE_CHECKING:
    from .flux_dev import FluxDevPipeline
    from .flux_kontext import FluxKontextPipeline
    from .flux2 import Flux2Pipeline
    from .flux2_diffusers import Flux2DiffusersPipeline
    from .flux2_klein import Flux2Klein4BPipeline, Flux2Klein9BPipeline
    from .flex1_alpha import Flex1AlphaPipeline
    from .flex2 import Flex2Pipeline
    from .sd15 import SD15Pipeline
    from .sdxl import SDXLPipeline
    from .qwen_image import (
        QwenImagePipeline,
        QwenImage2512Pipeline,
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline,
        QwenImageEditPlus2509Pipeline,
        QwenImageEditPlus2511Pipeline,
    )
    from .zimage import ZImagePipeline
    from .zimage_turbo import ZImageTurboPipeline
    from .zimage_deturbo import ZImageDeturboPipeline
    from .wan21 import (
        Wan21T2V14BPipeline,
        Wan21T2V1BPipeline,
        Wan21I2V14BPipeline,
        Wan21I2V14B480PPipeline,
    )
    from .wan22_t2v import Wan22T2V14BPipeline
    from .wan22_i2v import Wan22I2V14BPipeline
    from .wan22_5b import Wan22TI2V5BPipeline
    from .chroma import ChromaPipeline
    from .hidream import HiDreamPipeline, HiDreamE1Pipeline
    from .lumina2 import Lumina2Pipeline
    from .omnigen2 import OmniGen2Pipeline
    from .ltx2 import LTX2Pipeline


# Lazy import mapping: attribute name -> (module_name, class_name)
_LAZY_IMPORTS = {
    # FLUX family
    "FluxDevPipeline": (".flux_dev", "FluxDevPipeline"),
    "FluxKontextPipeline": (".flux_kontext", "FluxKontextPipeline"),
    "Flux2Pipeline": (".flux2", "Flux2Pipeline"),
    "Flux2DiffusersPipeline": (".flux2_diffusers", "Flux2DiffusersPipeline"),
    "Flux2Klein4BPipeline": (".flux2_klein", "Flux2Klein4BPipeline"),
    "Flux2Klein9BPipeline": (".flux2_klein", "Flux2Klein9BPipeline"),
    # Flex family
    "Flex1AlphaPipeline": (".flex1_alpha", "Flex1AlphaPipeline"),
    "Flex2Pipeline": (".flex2", "Flex2Pipeline"),
    # Stable Diffusion family
    "SD15Pipeline": (".sd15", "SD15Pipeline"),
    "SDXLPipeline": (".sdxl", "SDXLPipeline"),
    # Qwen family
    "QwenImagePipeline": (".qwen_image", "QwenImagePipeline"),
    "QwenImage2512Pipeline": (".qwen_image", "QwenImage2512Pipeline"),
    "QwenImageEditPipeline": (".qwen_image", "QwenImageEditPipeline"),
    "QwenImageEditPlusPipeline": (".qwen_image", "QwenImageEditPlusPipeline"),
    "QwenImageEditPlus2509Pipeline": (".qwen_image", "QwenImageEditPlus2509Pipeline"),
    "QwenImageEditPlus2511Pipeline": (".qwen_image", "QwenImageEditPlus2511Pipeline"),
    # Z-Image family
    "ZImagePipeline": (".zimage", "ZImagePipeline"),
    "ZImageTurboPipeline": (".zimage_turbo", "ZImageTurboPipeline"),
    "ZImageDeturboPipeline": (".zimage_deturbo", "ZImageDeturboPipeline"),
    # Wan 2.1 family
    "Wan21T2V14BPipeline": (".wan21", "Wan21T2V14BPipeline"),
    "Wan21T2V1BPipeline": (".wan21", "Wan21T2V1BPipeline"),
    "Wan21I2V14BPipeline": (".wan21", "Wan21I2V14BPipeline"),
    "Wan21I2V14B480PPipeline": (".wan21", "Wan21I2V14B480PPipeline"),
    # Wan 2.2 family
    "Wan22T2V14BPipeline": (".wan22_t2v", "Wan22T2V14BPipeline"),
    "Wan22I2V14BPipeline": (".wan22_i2v", "Wan22I2V14BPipeline"),
    "Wan22TI2V5BPipeline": (".wan22_5b", "Wan22TI2V5BPipeline"),
    # Chroma
    "ChromaPipeline": (".chroma", "ChromaPipeline"),
    # HiDream
    "HiDreamPipeline": (".hidream", "HiDreamPipeline"),
    "HiDreamE1Pipeline": (".hidream", "HiDreamE1Pipeline"),
    # Lumina
    "Lumina2Pipeline": (".lumina2", "Lumina2Pipeline"),
    # OmniGen
    "OmniGen2Pipeline": (".omnigen2", "OmniGen2Pipeline"),
    # LTX-2
    "LTX2Pipeline": (".ltx2", "LTX2Pipeline"),
}

# Cache for lazily loaded classes
_LAZY_CACHE = {}

_MODEL_TYPE_TO_CLASS = {
    # FLUX family
    ModelType.FLUX: "FluxDevPipeline",
    ModelType.FLUX_KONTEXT: "FluxKontextPipeline",
    ModelType.FLUX2: "Flux2Pipeline",
    ModelType.FLUX2_DIFFUSERS: "Flux2DiffusersPipeline",
    ModelType.FLUX2_KLEIN_4B: "Flux2Klein4BPipeline",
    ModelType.FLUX2_KLEIN_9B: "Flux2Klein9BPipeline",
    # Flex family
    ModelType.FLEX1: "Flex1AlphaPipeline",
    ModelType.FLEX2: "Flex2Pipeline",
    # Stable Diffusion family
    ModelType.SD15: "SD15Pipeline",
    ModelType.SDXL: "SDXLPipeline",
    # Qwen family
    ModelType.QWEN_IMAGE: "QwenImagePipeline",
    ModelType.QWEN_IMAGE_2512: "QwenImage2512Pipeline",
    ModelType.QWEN_IMAGE_EDIT: "QwenImageEditPipeline",
    ModelType.QWEN_IMAGE_EDIT_PLUS_2509: "QwenImageEditPlus2509Pipeline",
    ModelType.QWEN_IMAGE_EDIT_PLUS_2511: "QwenImageEditPlus2511Pipeline",
    # Z-Image family
    ModelType.ZIMAGE: "ZImagePipeline",
    ModelType.ZIMAGE_TURBO: "ZImageTurboPipeline",
    ModelType.ZIMAGE_DETURBO: "ZImageDeturboPipeline",
    # Wan 2.1 family
    ModelType.WAN21_14B: "Wan21T2V14BPipeline",
    ModelType.WAN21_1B: "Wan21T2V1BPipeline",
    ModelType.WAN21_I2V_14B: "Wan21I2V14BPipeline",
    ModelType.WAN21_I2V_14B_480P: "Wan21I2V14B480PPipeline",
    # Wan 2.2 family
    ModelType.WAN22_14B_T2V: "Wan22T2V14BPipeline",
    ModelType.WAN22_14B_I2V: "Wan22I2V14BPipeline",
    ModelType.WAN22_5B: "Wan22TI2V5BPipeline",
    # Chroma
    ModelType.CHROMA: "ChromaPipeline",
    # HiDream
    ModelType.HIDREAM: "HiDreamPipeline",
    ModelType.HIDREAM_E1: "HiDreamE1Pipeline",
    # Lumina
    ModelType.LUMINA2: "Lumina2Pipeline",
    # OmniGen
    ModelType.OMNIGEN2: "OmniGen2Pipeline",
    # LTX-2
    ModelType.LTX2: "LTX2Pipeline",
}


def __getattr__(name: str):
    """Lazy import handler - only loads pipeline classes when accessed."""
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    
    if name in _LAZY_IMPORTS:
        module_name, class_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_name, package=__name__)
        cls = getattr(module, class_name)
        _LAZY_CACHE[name] = cls
        return cls
    
    if name == "PIPELINE_REGISTRY":
        # Build registry lazily
        registry = _build_pipeline_registry()
        _LAZY_CACHE["PIPELINE_REGISTRY"] = registry
        return registry
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_pipeline_registry() -> dict:
    """Build the pipeline registry lazily."""
    # Import classes lazily as needed
    return {model_type: __getattr__(class_name) for model_type, class_name in _MODEL_TYPE_TO_CLASS.items()}


def get_pipeline_class(model: str) -> Optional[Type[BasePipeline]]:
    """Get pipeline class for a model type string."""
    try:
        model_type = ModelType(model)
    except ValueError:
        return None
    class_name = _MODEL_TYPE_TO_CLASS.get(model_type)
    if not class_name:
        return None
    return __getattr__(class_name)


def get_pipeline_config(model: str) -> Optional[PipelineConfig]:
    """Get pipeline config for a model type string."""
    pipeline_class = get_pipeline_class(model)
    if pipeline_class:
        return pipeline_class.CONFIG
    return None


__all__ = [
    # Base
    "BasePipeline",
    "PipelineConfig",
    "LoraMergeMethod",
    # FLUX family
    "FluxDevPipeline",
    "FluxKontextPipeline",
    "Flux2Pipeline",
    "Flux2DiffusersPipeline",
    "Flux2Klein4BPipeline",
    "Flux2Klein9BPipeline",
    # Flex family
    "Flex1AlphaPipeline",
    "Flex2Pipeline",
    # Stable Diffusion family
    "SD15Pipeline",
    "SDXLPipeline",
    # Qwen family
    "QwenImagePipeline",
    "QwenImage2512Pipeline",
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
    "QwenImageEditPlus2509Pipeline",
    "QwenImageEditPlus2511Pipeline",
    # Z-Image family
    "ZImagePipeline",
    "ZImageTurboPipeline",
    "ZImageDeturboPipeline",
    # Wan 2.1 family
    "Wan21T2V14BPipeline",
    "Wan21T2V1BPipeline",
    "Wan21I2V14BPipeline",
    "Wan21I2V14B480PPipeline",
    # Wan 2.2 family
    "Wan22T2V14BPipeline",
    "Wan22I2V14BPipeline",
    "Wan22TI2V5BPipeline",
    # Chroma
    "ChromaPipeline",
    # HiDream
    "HiDreamPipeline",
    "HiDreamE1Pipeline",
    # Lumina
    "Lumina2Pipeline",
    # OmniGen
    "OmniGen2Pipeline",
    # LTX-2
    "LTX2Pipeline",
    # Registry and helpers
    "get_pipeline_class",
    "get_pipeline_config",
    "PIPELINE_REGISTRY",
]
