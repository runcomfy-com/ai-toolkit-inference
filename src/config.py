"""
Configuration management for inference server.
"""

import os
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


def _get_default_ai_toolkit_path() -> str:
    """
    Determine the default ai-toolkit path.

    Priority:
    1. AI_TOOLKIT_PATH environment variable (if set)
    2. Vendored ai-toolkit at <repo>/vendor/ai-toolkit (if exists)
    3. Fallback to /app/ai-toolkit
    """
    # 1. Check env var (handled by pydantic, but we compute fallback here)
    env_val = os.environ.get("AI_TOOLKIT_PATH")
    if env_val:
        return env_val

    # 2. Check vendored location (installed by install.py / ComfyUI-Manager)
    # This file is at <repo>/src/config.py, so repo root is one level up
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendored_path = os.path.join(repo_root, "vendor", "ai-toolkit")
    if os.path.isdir(vendored_path):
        return vendored_path

    # 3. Fallback
    return "/app/ai-toolkit"


class Settings(BaseSettings):
    """Server settings loaded from environment variables."""

    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Base URL for generating status/result URLs
    base_url: str = Field(default="http://localhost:8000", alias="BASE_URL")

    # Paths configuration
    # Workflow directory: {workflows_base_path}/{lora_name}/
    # Contains: config.yml and {lora_name}.safetensors
    workflows_base_path: str = Field(default="/app/ai-toolkit/lora_weights", alias="WORKFLOWS_BASE_PATH")

    # Output directory for inference results
    # Results are stored as: {output_base_path}/{request_id}/
    output_base_path: str = Field(default="/tmp/inference_output", alias="OUTPUT_BASE_PATH")

    # Device settings
    device: str = Field(default="cuda", alias="DEVICE")

    # CPU offload mode: "none", "model", or "sequential"
    # - "none": No offload, model stays on GPU (fastest inference, highest VRAM)
    # - "model": Model CPU offload - moves full model between CPU/GPU (balanced)
    # - "sequential": Sequential CPU offload - moves layers one at a time (lowest VRAM, slowest)
    offload_mode: Literal["none", "model", "sequential"] = Field(default="none", alias="OFFLOAD_MODE")

    # Deprecated: use offload_mode instead. Kept for backwards compatibility.
    # If set to True and offload_mode is not explicitly set, will use "model" mode.
    enable_cpu_offload: bool = Field(default=False, alias="ENABLE_CPU_OFFLOAD")

    # Model cache settings
    model_cache_dir: str | None = Field(default=None, alias="MODEL_CACHE_DIR")

    # HuggingFace token for gated models (optional)
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # Inference timeout in seconds
    inference_timeout: int = Field(default=3600, alias="INFERENCE_TIMEOUT")

    # AI-Toolkit path for extended model support (Chroma, FLUX.2, HiDream, OmniGen2, Wan2.2, LTX2)
    # These models require custom pipelines from ai-toolkit.
    # Auto-detected: prefers AI_TOOLKIT_PATH env var, else vendor/ai-toolkit if present.
    ai_toolkit_path: str = Field(default_factory=_get_default_ai_toolkit_path, alias="AI_TOOLKIT_PATH")

    # LoRA download cache directory for URL-based LoRA files
    lora_download_cache_dir: str = Field(default="/tmp/lora_cache", alias="LORA_DOWNLOAD_CACHE_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


# Global settings instance
settings = Settings()
