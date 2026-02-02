"""
Latent contract mixin for Qwen Image (mean/std normalized, spatial latents).
"""

from .base import LatentContractMixin


class QwenLatentMixin(LatentContractMixin):
    """
    Qwen Image exposes spatial latents (not packed) normalized by mean/std.
    """

    LATENT_NORM = "qwen_mean_std"
    LATENT_PACKED = False
