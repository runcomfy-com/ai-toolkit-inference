"""
Shared helpers for ai-toolkit-inference ComfyUI nodes.

Design goals:
- Keep ComfyUI node import lightweight (avoid importing heavy pipelines until execution).
- Provide consistent tensor<->PIL conversions.
- Cache a single loaded pipeline instance to avoid reload cost and GPU churn.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def comfy_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor to PIL.

    ComfyUI convention: [B, H, W, C] float32 in [0, 1].
    We take the first image in the batch.
    """
    if tensor is None:
        raise ValueError("control image tensor is None")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 4 or tensor.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(tensor.shape)}")

    img = tensor[0].detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    if img.shape[-1] == 1:
        img = img[..., 0]
        return Image.fromarray(img, mode="L").convert("RGB")
    if img.shape[-1] == 4:
        return Image.fromarray(img, mode="RGBA").convert("RGB")
    return Image.fromarray(img, mode="RGB")


def pil_to_comfy_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor [B,H,W,C] float32 in [0,1]."""
    arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1, H, W, 3]


def pil_frames_to_comfy_images(frames: List[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL frames to a ComfyUI IMAGE batch.

    Returns tensor [T, H, W, 3], where T is number of frames.
    """
    if not frames:
        raise ValueError("frames list is empty")
    arrs = []
    for fr in frames:
        arr = np.array(fr.convert("RGB")).astype(np.float32) / 255.0
        arrs.append(arr)
    batch = np.stack(arrs, axis=0)  # [T, H, W, 3]
    return torch.from_numpy(batch)


def _normalize_lora_paths_for_cache(lora_paths: List[Union[str, Dict[str, str]]]) -> Tuple:
    """Turn lora_paths into a hashable key.

    Supports string paths and MoE dict format (e.g. {"high": "...", "low": "..."}).
    """
    norm: List[Any] = []
    for p in (lora_paths or []):
        if isinstance(p, dict):
            norm.append(tuple(sorted((k, v) for k, v in p.items() if v)))
        else:
            norm.append(str(p))
    return tuple(norm)


def _normalize_lora_scale_for_cache(
    lora_scale: Union[float, Dict[str, float]],
) -> Union[float, Tuple[Tuple[str, float], ...]]:
    """Turn lora_scale into a hashable key (supports MoE dict)."""
    if isinstance(lora_scale, dict):
        return tuple(sorted((k, float(v)) for k, v in lora_scale.items()))
    return float(lora_scale)


def _normalize_lora_scale_value(
    lora_scale: Union[float, Dict[str, float]],
) -> Union[float, Dict[str, float]]:
    """Normalize lora_scale to concrete float values for pipeline loading."""
    if isinstance(lora_scale, dict):
        return {k: float(v) for k, v in lora_scale.items() if v is not None}
    return float(lora_scale)


def _maybe_download_lora_paths(lora_paths: List[Union[str, Dict[str, str]]]) -> List[Union[str, Dict[str, str]]]:
    """Download URL-based LoRAs to a local cache dir.

    This allows pipelines to keep using local_files_only=True.
    """
    if not lora_paths:
        return []

    try:
        from src.services.pipeline_manager import is_url, _download_lora_from_url  # type: ignore
    except Exception:
        return lora_paths

    cache_dir = os.environ.get("AITK_LORA_CACHE_DIR", "/tmp/lora_cache")

    resolved: List[Union[str, Dict[str, str]]] = []
    for p in lora_paths:
        if isinstance(p, dict):
            d: Dict[str, str] = {}
            for k, v in p.items():
                if not v:
                    continue
                d[k] = _download_lora_from_url(v, cache_dir) if is_url(v) else v
            resolved.append(d)
        else:
            resolved.append(_download_lora_from_url(p, cache_dir) if is_url(p) else p)
    return resolved


@dataclass(frozen=True)
class PipelineCacheKey:
    model_id: str
    pipeline_id: str  # e.g. "module:ClassName" to distinguish different pipeline classes
    hf_token: Optional[str]
    lora_paths_key: Tuple
    lora_scale_key: Union[float, Tuple[Tuple[str, float], ...]]


def _get_pipeline_id(pipeline_ctor) -> str:
    """Get a unique identifier for a pipeline class/constructor."""
    if hasattr(pipeline_ctor, "__module__") and hasattr(pipeline_ctor, "__name__"):
        return f"{pipeline_ctor.__module__}:{pipeline_ctor.__name__}"
    return str(pipeline_ctor)


_PIPELINE_CACHE: Dict[str, Any] = {
    "key": None,
    "instance": None,
}


def get_or_load_pipeline(
    *,
    model_id: str,
    pipeline_ctor,
    enable_cpu_offload: bool,
    hf_token: Optional[str],
    lora_paths: List[Union[str, Dict[str, str]]],
    lora_scale: Union[float, Dict[str, float]],
) -> Any:
    """Load and cache a single pipeline instance.

    If model/token/lora/pipeline_class config changes, unload old pipeline and load new.
    """
    global _PIPELINE_CACHE

    resolved_loras = _maybe_download_lora_paths(lora_paths)
    scale_value = _normalize_lora_scale_value(lora_scale)
    pipeline_id = _get_pipeline_id(pipeline_ctor)
    key = PipelineCacheKey(
        model_id=model_id,
        pipeline_id=pipeline_id,
        hf_token=hf_token or None,
        lora_paths_key=_normalize_lora_paths_for_cache(resolved_loras),
        lora_scale_key=_normalize_lora_scale_for_cache(scale_value),
    )

    if _PIPELINE_CACHE["instance"] is not None and _PIPELINE_CACHE["key"] == key:
        return _PIPELINE_CACHE["instance"]

    if _PIPELINE_CACHE["instance"] is not None:
        try:
            _PIPELINE_CACHE["instance"].unload()
        except Exception:
            pass
        _PIPELINE_CACHE["instance"] = None
        _PIPELINE_CACHE["key"] = None

    logger.info(f"Loading pipeline model_id={model_id} loras={resolved_loras} scale={scale_value}")
    pipe = pipeline_ctor(device="cuda", enable_cpu_offload=enable_cpu_offload, hf_token=hf_token or None)
    pipe.load(lora_paths=resolved_loras, lora_scale=scale_value)

    _PIPELINE_CACHE["instance"] = pipe
    _PIPELINE_CACHE["key"] = key
    return pipe
