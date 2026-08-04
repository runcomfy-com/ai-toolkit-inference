"""
Centralized model download configuration (allow/ignore patterns + extra repos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..schemas.models import ModelType


@dataclass(frozen=True)
class ExtraDownload:
    """Extra repo to download alongside a base model."""

    repo_id: str
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None


@dataclass(frozen=True)
class DownloadConfig:
    """Download filters + extra repos for a model type."""

    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    extras: List[ExtraDownload] = field(default_factory=list)


FP16_SD_PATTERNS = ["*.fp16.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"]


MODEL_DOWNLOAD_CONFIGS: Dict[ModelType, DownloadConfig] = {
    # FLUX family
    ModelType.FLUX: DownloadConfig(),
    ModelType.FLUX_KONTEXT: DownloadConfig(),
    ModelType.FLUX2: DownloadConfig(
        allow_patterns=["flux2-dev.safetensors", "ae.safetensors"],
        extras=[ExtraDownload(repo_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503")],
    ),
    ModelType.FLUX2_DIFFUSERS: DownloadConfig(),
    ModelType.FLUX2_KLEIN_4B: DownloadConfig(
        allow_patterns=["flux-2-klein-base-4b.safetensors"],
        extras=[
            ExtraDownload(repo_id="Qwen/Qwen3-4B"),
            ExtraDownload(repo_id="ai-toolkit/flux2_vae", allow_patterns=["ae.safetensors"]),
        ],
    ),
    ModelType.FLUX2_KLEIN_9B: DownloadConfig(
        allow_patterns=["flux-2-klein-base-9b.safetensors"],
        extras=[
            ExtraDownload(repo_id="Qwen/Qwen3-8B"),
            ExtraDownload(repo_id="ai-toolkit/flux2_vae", allow_patterns=["ae.safetensors"]),
        ],
    ),
    # Flex family
    ModelType.FLEX1: DownloadConfig(),
    ModelType.FLEX2: DownloadConfig(),
    # Stable Diffusion family
    ModelType.SD15: DownloadConfig(allow_patterns=FP16_SD_PATTERNS),
    ModelType.SDXL: DownloadConfig(allow_patterns=FP16_SD_PATTERNS),
    # Qwen family
    ModelType.QWEN_IMAGE: DownloadConfig(),
    ModelType.QWEN_IMAGE_2512: DownloadConfig(),
    ModelType.QWEN_IMAGE_EDIT: DownloadConfig(),
    ModelType.QWEN_IMAGE_EDIT_PLUS_2509: DownloadConfig(),
    ModelType.QWEN_IMAGE_EDIT_PLUS_2511: DownloadConfig(),
    # Z-Image family
    ModelType.ZIMAGE: DownloadConfig(),
    ModelType.ZIMAGE_TURBO: DownloadConfig(),
    ModelType.ZIMAGE_DETURBO: DownloadConfig(),
    # Wan 2.1 family
    ModelType.WAN21_14B: DownloadConfig(),
    ModelType.WAN21_1B: DownloadConfig(),
    ModelType.WAN21_I2V_14B: DownloadConfig(),
    ModelType.WAN21_I2V_14B_480P: DownloadConfig(),
    # Wan 2.2 family
    ModelType.WAN22_14B_T2V: DownloadConfig(ignore_patterns=["transformer/*", "transformer_2/*"]),
    ModelType.WAN22_14B_I2V: DownloadConfig(
        extras=[
            ExtraDownload(repo_id="ai-toolkit/wan2.1-vae"),
            ExtraDownload(
                repo_id="ai-toolkit/umt5_xxl_encoder",
                allow_patterns=["text_encoder/*", "tokenizer/*"],
            ),
        ]
    ),
    ModelType.WAN22_5B: DownloadConfig(),
    # Chroma
    ModelType.CHROMA: DownloadConfig(
        allow_patterns=["Chroma1-Base.safetensors"],
        extras=[
            ExtraDownload(
                repo_id="ostris/Flex.1-alpha",
                allow_patterns=["vae/*", "text_encoder_2/*", "tokenizer_2/*"],
            )
        ],
    ),
    # HiDream
    ModelType.HIDREAM: DownloadConfig(
        extras=[ExtraDownload(repo_id="unsloth/Meta-Llama-3.1-8B-Instruct")]
    ),
    ModelType.HIDREAM_E1: DownloadConfig(
        extras=[ExtraDownload(repo_id="unsloth/Meta-Llama-3.1-8B-Instruct")]
    ),
    # Lumina
    ModelType.LUMINA2: DownloadConfig(),
    # OmniGen
    ModelType.OMNIGEN2: DownloadConfig(),
    # LTX-2
    ModelType.LTX2: DownloadConfig(
        ignore_patterns=[
            "ltx-2-*.safetensors",
            "latent_upsampler/*",
            "*.mp4",
            "*.gguf",
            "text_encoder/diffusion_pytorch_model*",
        ]
    ),
    # LTX-2.3
    # Note: Spatial upsampler and distilled LoRA are NOT predownloaded here.
    # They are lazy-loaded on first High-mode request in LTX23Pipeline to avoid
    # making them unconditional dependencies for Default-mode requests.
    ModelType.LTX2_3: DownloadConfig(),
    # Krea 2
    # Only the flat single-file checkpoint at the repo root is loaded. The krea/*
    # repos ALSO ship a full diffusers folder layout (the same weights again, and
    # a copy of the text encoder / VAE), so allow_patterns avoids a duplicate
    # ~26 GB download per model.
    # NOTE: every entry sharing a repo must declare identical allow_patterns --
    # _merge_download_tasks() in cli/download_models.py dedupes by repo_id and
    # collapses to an unfiltered full-repo pull if either side is None.
    # NOTE: krea/Krea-2-* are HF gated ("auto"); HF_TOKEN must belong to an
    # account that has accepted the Krea 2 license.
    ModelType.KREA2: DownloadConfig(
        allow_patterns=["raw.safetensors"],
        extras=[
            ExtraDownload(repo_id="Qwen/Qwen3-VL-4B-Instruct"),
            ExtraDownload(repo_id="Qwen/Qwen-Image", allow_patterns=["vae/*"]),
        ],
    ),
    ModelType.KREA2_TURBO: DownloadConfig(
        allow_patterns=["turbo.safetensors"],
        extras=[
            ExtraDownload(repo_id="Qwen/Qwen3-VL-4B-Instruct"),
            ExtraDownload(repo_id="Qwen/Qwen-Image", allow_patterns=["vae/*"]),
        ],
    ),
    ModelType.KREA2_O_EDIT: DownloadConfig(
        allow_patterns=["raw.safetensors"],
        extras=[
            ExtraDownload(repo_id="Qwen/Qwen3-VL-4B-Instruct"),
            ExtraDownload(repo_id="Qwen/Qwen-Image", allow_patterns=["vae/*"]),
        ],
    ),
    ModelType.KREA2_O_EDIT_TURBO: DownloadConfig(
        allow_patterns=["turbo.safetensors"],
        extras=[
            ExtraDownload(repo_id="Qwen/Qwen3-VL-4B-Instruct"),
            ExtraDownload(repo_id="Qwen/Qwen-Image", allow_patterns=["vae/*"]),
        ],
    ),

    # MiniMax-H3 (fl2va). allow_patterns is NOT optional here: unfiltered,
    # Comfy-Org/MiniMax-H3 is ~365 GB (bf16 + int8 + fp8 + int8-pruned, two
    # partitions, three text encoders). These four files are 42.5 GB.
    # Likewise MiniMaxAI/MiniMax-H3 is 297 files including the full bf16
    # original -- we want three tiny config/tokenizer subfolders.
    # Watch the _merge_download_tasks dedupe invariant
    # (src/cli/download_models.py:51-69): if either side's patterns are None
    # the merge collapses to a full pull. No other entry shares these repos.
    ModelType.MINIMAX_H3: DownloadConfig(
        allow_patterns=[
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        ],
        extras=[
            ExtraDownload(
                repo_id="MiniMaxAI/MiniMax-H3",
                allow_patterns=[
                    "FL2VA/tokenizer/*",
                    "FL2VA/processor/*",
                    "FL2VA/text_encoder/config.json",
                ],
            ),
        ],
    ),
}


def get_download_config(model_type: ModelType) -> DownloadConfig:
    """Return download config for a model type (defaults to empty config)."""
    return MODEL_DOWNLOAD_CONFIGS.get(model_type, DownloadConfig())
