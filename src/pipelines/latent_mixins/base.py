"""
Latent contract mixins for pipeline interoperability (ComfyUI-friendly).
"""

from typing import Any, Dict, Optional


class LatentContractMixin:
    """
    Provides a consistent latent metadata contract.

    Pipelines may override class attributes or helper methods to customize.
    """

    # Human-readable latent normalization identifier
    LATENT_NORM: str = "unknown"
    # Whether the exposed latents are packed (ComfyUI expects spatial)
    LATENT_PACKED: bool = False

    def _latent_channels_default(self) -> Optional[int]:
        """Best-effort latent channel count."""
        pipe = getattr(self, "pipe", None)
        vae = getattr(pipe, "vae", None) if pipe is not None else None
        try:
            if vae is not None and getattr(vae, "config", None) is not None:
                return int(getattr(vae.config, "latent_channels", 0) or 0) or None
        except Exception:
            pass
        return None

    def _vae_scale_factor_default(self) -> Optional[int]:
        """Best-effort VAE scale factor (spatial downscale)."""
        pipe = getattr(self, "pipe", None)
        try:
            if pipe is not None and hasattr(pipe, "vae_scale_factor"):
                return int(getattr(pipe, "vae_scale_factor", 0) or 0) or None
        except Exception:
            pass
        return None

    def latent_metadata(
        self,
        latents=None,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build a ComfyUI-friendly latent metadata dict.

        Intended to be attached to LATENT dicts alongside "samples".
        """
        meta: Dict[str, Any] = {}

        # Model type (stable across pipelines)
        cfg = getattr(self, "CONFIG", None)
        model_type = getattr(cfg, "model_type", None)
        if model_type is not None:
            # Enum values are user-friendly; fall back to str() if needed
            meta["aitk_model_type"] = getattr(model_type, "value", str(model_type))

        # Latent shape hints
        channels = None
        if latents is not None:
            try:
                if getattr(latents, "ndim", 0) >= 2:
                    channels = int(latents.shape[1])
            except Exception:
                channels = None
        if channels is None:
            channels = self._latent_channels_default()
        if channels is not None:
            meta["aitk_latent_channels"] = int(channels)

        meta["aitk_latent_norm"] = self.LATENT_NORM
        meta["aitk_packed"] = bool(self.LATENT_PACKED)

        vae_sf = self._vae_scale_factor_default()
        if vae_sf is not None:
            meta["aitk_vae_scale_factor"] = int(vae_sf)

        if width is not None:
            meta["aitk_width"] = int(width)
        if height is not None:
            meta["aitk_height"] = int(height)

        return meta
