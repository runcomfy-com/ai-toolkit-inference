"""
Pipeline implementations for different models.

Each pipeline defines its own CONFIG (PipelineConfig) as a class attribute.
The PIPELINE_REGISTRY maps ModelType values to pipeline classes.
"""

from typing import Type, Optional

from .base import BasePipeline, PipelineConfig, LoraMergeMethod

# FLUX family
from .flux_dev import FluxDevPipeline
from .flux_kontext import FluxKontextPipeline
from .flux2 import Flux2Pipeline
from .flux2_diffusers import Flux2DiffusersPipeline

# Flex family
from .flex1_alpha import Flex1AlphaPipeline
from .flex2 import Flex2Pipeline

# Stable Diffusion family
from .sd15 import SD15Pipeline
from .sdxl import SDXLPipeline

# Qwen family
from .qwen_image import (
    QwenImagePipeline,
    QwenImage2512Pipeline,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
    QwenImageEditPlus2509Pipeline,
    QwenImageEditPlus2511Pipeline,
)

# Z-Image family
from .zimage_turbo import ZImageTurboPipeline
from .zimage_deturbo import ZImageDeturboPipeline

# Wan 2.1 family
from .wan21 import (
    Wan21T2V14BPipeline,
    Wan21T2V1BPipeline,
    Wan21I2V14BPipeline,
    Wan21I2V14B480PPipeline,
)

# Wan 2.2 family
from .wan22_t2v import Wan22T2V14BPipeline
from .wan22_i2v import Wan22I2V14BPipeline
from .wan22_5b import Wan22TI2V5BPipeline

# Chroma
from .chroma import ChromaPipeline

# HiDream
from .hidream import HiDreamPipeline, HiDreamE1Pipeline

# Lumina
from .lumina2 import Lumina2Pipeline

# OmniGen
from .omnigen2 import OmniGen2Pipeline

# LTX-2
from .ltx2 import LTX2Pipeline

from ..schemas.models import ModelType


# Pipeline registry: maps ModelType to pipeline class
# Each pipeline class has its own CONFIG (PipelineConfig)
PIPELINE_REGISTRY: dict[ModelType, Type[BasePipeline]] = {
    # FLUX family
    ModelType.FLUX: FluxDevPipeline,
    ModelType.FLUX_KONTEXT: FluxKontextPipeline,
    ModelType.FLUX2: Flux2Pipeline,
    ModelType.FLUX2_DIFFUSERS: Flux2DiffusersPipeline,
    # Flex family
    ModelType.FLEX1: Flex1AlphaPipeline,
    ModelType.FLEX2: Flex2Pipeline,
    # Stable Diffusion family
    ModelType.SD15: SD15Pipeline,
    ModelType.SDXL: SDXLPipeline,
    # Qwen family
    ModelType.QWEN_IMAGE: QwenImagePipeline,
    ModelType.QWEN_IMAGE_2512: QwenImage2512Pipeline,
    ModelType.QWEN_IMAGE_EDIT: QwenImageEditPipeline,
    ModelType.QWEN_IMAGE_EDIT_PLUS_2509: QwenImageEditPlus2509Pipeline,
    ModelType.QWEN_IMAGE_EDIT_PLUS_2511: QwenImageEditPlus2511Pipeline,
    # Z-Image family
    ModelType.ZIMAGE_TURBO: ZImageTurboPipeline,
    ModelType.ZIMAGE_DETURBO: ZImageDeturboPipeline,
    # Wan 2.1 family
    ModelType.WAN21_14B: Wan21T2V14BPipeline,
    ModelType.WAN21_1B: Wan21T2V1BPipeline,
    ModelType.WAN21_I2V_14B: Wan21I2V14BPipeline,
    ModelType.WAN21_I2V_14B_480P: Wan21I2V14B480PPipeline,
    # Wan 2.2 family
    ModelType.WAN22_14B_T2V: Wan22T2V14BPipeline,
    ModelType.WAN22_14B_I2V: Wan22I2V14BPipeline,
    ModelType.WAN22_5B: Wan22TI2V5BPipeline,
    # Chroma
    ModelType.CHROMA: ChromaPipeline,
    # HiDream
    ModelType.HIDREAM: HiDreamPipeline,
    ModelType.HIDREAM_E1: HiDreamE1Pipeline,
    # Lumina
    ModelType.LUMINA2: Lumina2Pipeline,
    # OmniGen
    ModelType.OMNIGEN2: OmniGen2Pipeline,
    # LTX-2
    ModelType.LTX2: LTX2Pipeline,
}


def get_pipeline_class(model: str) -> Optional[Type[BasePipeline]]:
    """Get pipeline class for a model type string."""
    try:
        model_type = ModelType(model)
        return PIPELINE_REGISTRY.get(model_type)
    except ValueError:
        return None


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
