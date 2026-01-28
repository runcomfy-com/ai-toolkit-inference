"""
Response schemas for inference API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ImageOutput(BaseModel):
    """Single image output."""

    format: str = Field(default="jpeg", description="Image format (jpeg, png)")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    file_path: str = Field(..., description="Local file path")
    seed: int = Field(..., description="Seed used for generation")


class VideoOutput(BaseModel):
    """Single video output."""

    format: str = Field(default="webp", description="Video format (webp, mp4)")
    width: int = Field(..., description="Video width")
    height: int = Field(..., description="Video height")
    num_frames: int = Field(..., description="Number of frames")
    fps: int = Field(..., description="Frames per second")
    file_path: str = Field(..., description="Local file path")
    seed: int = Field(..., description="Seed used for generation")


class InferenceResponse(BaseModel):
    """Response for POST /v1/inference."""

    request_id: str = Field(..., description="Unique request ID")
    status: str = Field(default="queued", description="Initial status")
    status_url: str = Field(..., description="URL to check request status")
    result_url: str = Field(..., description="URL to get request result")
    created_at: str = Field(..., description="Request creation timestamp (ISO format)")


class InferenceStatusResponse(BaseModel):
    """Response for GET /v1/requests/{request_id}/status."""

    request_id: str = Field(..., description="Unique request ID")
    status: str = Field(..., description="Request status: in_queue, in_progress, succeeded, failed, canceled")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    started_at: Optional[str] = Field(default=None, description="Processing start timestamp")
    finished_at: Optional[str] = Field(default=None, description="Completion timestamp")
    error: Optional[str] = Field(default=None, description="Error message (only if failed)")


class InferenceResultResponse(BaseModel):
    """Response for GET /v1/requests/{request_id}/result."""

    request_id: str = Field(..., description="Unique request ID")
    status: str = Field(..., description="Request status")
    outputs: Optional[Dict[str, Any]] = Field(default=None, description="Inference outputs")
    error: Optional[str] = Field(default=None, description="Error message (only if failed)")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    finished_at: Optional[str] = Field(default=None, description="Completion timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Inference metadata")


class ErrorResponse(BaseModel):
    """Error response format."""

    error: Dict[str, Any] = Field(..., description="Error details")

    class Config:
        json_schema_extra = {
            "example": {"error": {"code": "INVALID_MODEL", "message": "Model 'xxx' is not supported", "details": {}}}
        }
