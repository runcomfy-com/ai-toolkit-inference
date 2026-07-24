"""
Request schemas for inference API.

Parameter names match inference_schema JSON files exactly.
"""

from typing import Optional, List, Literal, get_args
from pydantic import BaseModel, Field, model_validator


class LoraItem(BaseModel):
    """LoRA item for inference requests."""

    path: str = Field(..., description="LoRA file path or URL")
    transformer: Optional[Literal["low", "high"]] = Field(
        default=None,
        description="MoE transformer selector (low/high). Only for MoE models.",
    )
    network_multiplier: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="LoRA Scale",
    )


class PromptItem(BaseModel):
    """
    Single prompt item for batch inference.

    Each item can override global parameters with per-prompt values.
    All parameters match inference_schema JSON files.
    """

    prompt: str = Field(..., description="The prompt text")
    trigger_word: Optional[str] = Field(default=None, description="Trigger word to replace [trigger] in this prompt")

    # Image parameters (can override global values)
    width: Optional[int] = Field(default=None, ge=64, le=4096, description="Width")
    height: Optional[int] = Field(default=None, ge=64, le=4096, description="Height")
    guidance_scale: Optional[float] = Field(default=None, ge=0, le=20, description="Guidance Scale")
    sample_steps: Optional[int] = Field(default=None, ge=1, le=150, description="Sample Steps")
    neg: Optional[str] = Field(default=None, description="Negative Prompt")
    seed: Optional[int] = Field(default=None, ge=-1, description="Seed")
    sampler: Optional[str] = Field(default=None, description="Sampler")

    # Video parameters
    num_frames: Optional[int] = Field(default=None, ge=1, le=1000, description="Num Frames")
    fps: Optional[int] = Field(default=None, ge=1, le=120, description="FPS")

    # Resolution mode (Default or High for 2x upscale, LTX-2.3 only)
    resolution: Optional[Literal["Default", "High"]] = Field(
        default=None, description="Resolution mode: Default or High (2x upscale)"
    )

    # Control images (matches inference_schema)
    ctrl_img: Optional[str] = Field(default=None, description="Control image as base64 or URL")
    ctrl_img_1: Optional[str] = Field(default=None, description="Control image 1")
    ctrl_img_2: Optional[str] = Field(default=None, description="Control image 2")
    ctrl_img_3: Optional[str] = Field(default=None, description="Control image 3")

    def get_control_image(self) -> Optional[str]:
        """Get single control image."""
        return self.ctrl_img

    def get_control_images(self) -> List[str]:
        """Get all control images as a list."""
        images = []
        if self.ctrl_img_1:
            images.append(self.ctrl_img_1)
        if self.ctrl_img_2:
            images.append(self.ctrl_img_2)
        if self.ctrl_img_3:
            images.append(self.ctrl_img_3)

        # Fallback to single ctrl_img
        if not images and self.ctrl_img:
            images = [self.ctrl_img]

        return images


class InferenceInput(BaseModel):
    """
    Input payload for inference requests.

    All image/video parameters are specified in each PromptItem.
    """

    # Required parameters
    model: str = Field(..., description="Model identifier (e.g., flux, flex2, wan22_5b)")
    loras: List[LoraItem] = Field(..., description="List of LoRA items")

    # HuggingFace token (priority: API request > environment variable)
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token for gated models. Priority: API request > HF_TOKEN env var",
    )

    # Trigger word for prompt replacement
    trigger_word: Optional[str] = Field(
        default=None,
        description="Trigger word to replace [trigger] placeholder in prompts",
    )

    # Prompts (required, all parameters are in PromptItem)
    prompts: List[PromptItem] = Field(..., description="List of prompts with their parameters")

    # --- Model-specific options -------------------------------------------
    # These must match how the LoRA was TRAINED and cannot be read back from the
    # adapter file (ai-toolkit's LoRA metadata does not record model_kwargs).
    # Ignored by models that do not use them.
    kv_cache: Optional[bool] = Field(
        default=None,
        description=(
            "Krea 2 edit only. Must match the training config's "
            "model_kwargs.kv_cache -- it changes the attention mask, so a "
            "mismatch silently produces wrong output. Defaults to the current "
            "ai-toolkit preset (true); set false for adapters trained before "
            "2026-07-16."
        ),
    )
    match_target_res: Optional[bool] = Field(
        default=None,
        description=(
            "Krea 2 edit only. Must match the training config's "
            "model_kwargs.match_target_res. When true, each reference image is "
            "area-matched to the target resolution; when false it is only "
            "capped at 1 MP. Defaults to the current preset (true)."
        ),
    )

    def get_model_options(self) -> dict:
        """Model-specific options to hand to BasePipeline.apply_model_options()."""
        options = {}
        if self.kv_cache is not None:
            options["kv_cache"] = self.kv_cache
        if self.match_target_res is not None:
            options["match_target_res"] = self.match_target_res
        return options

    @model_validator(mode="after")
    def validate_prompts(self):
        """Ensure at least one prompt is provided."""
        if not self.prompts or len(self.prompts) == 0:
            raise ValueError("At least one prompt must be provided in 'prompts'")
        return self

    def get_prompts(self) -> List[PromptItem]:
        """Get all prompts as a list of PromptItem."""
        return self.prompts
