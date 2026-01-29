"""
FLUX.2-klein pipeline implementations.
"""

import logging

from .base import PipelineConfig, LoraMergeMethod
from .flux2 import Flux2Pipeline
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)


class Flux2Klein4BPipeline(Flux2Pipeline):
    """
    FLUX.2-klein 4B pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLUX2_KLEIN_4B,
        base_model="black-forest-labs/FLUX.2-klein-base-4B",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        lora_merge_method=LoraMergeMethod.CUSTOM,
    )

    TRANSFORMER_FILENAME = "flux-2-klein-base-4b.safetensors"
    TEXT_ENCODER_REPO = "Qwen/Qwen3-4B"
    TEXT_ENCODER_TYPE = "qwen"
    VAE_REPO = "ai-toolkit/flux2_vae"
    IS_GUIDANCE_DISTILLED = False

    def _get_flux2_params(self):
        from extensions_built_in.diffusion_models.flux2.src.model import Klein4BParams
        return Klein4BParams()


class Flux2Klein9BPipeline(Flux2Pipeline):
    """
    FLUX.2-klein 9B pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLUX2_KLEIN_9B,
        base_model="black-forest-labs/FLUX.2-klein-base-9B",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        lora_merge_method=LoraMergeMethod.CUSTOM,
    )

    TRANSFORMER_FILENAME = "flux-2-klein-base-9b.safetensors"
    TEXT_ENCODER_REPO = "Qwen/Qwen3-8B"
    TEXT_ENCODER_TYPE = "qwen"
    VAE_REPO = "ai-toolkit/flux2_vae"
    IS_GUIDANCE_DISTILLED = False

    def _get_flux2_params(self):
        from extensions_built_in.diffusion_models.flux2.src.model import Klein9BParams
        return Klein9BParams()
