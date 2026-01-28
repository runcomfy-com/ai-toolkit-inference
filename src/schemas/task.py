"""
Task schemas for internal request management.
"""

from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class RequestStatus(str, Enum):
    """Request status enum."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def to_api_status(self) -> str:
        """Convert to API-facing status."""
        mapping = {
            "queued": "in_queue",
            "processing": "in_progress",
            "succeeded": "succeeded",
            "failed": "failed",
            "cancelled": "canceled",
        }
        return mapping.get(self.value, self.value)


class InferenceTask(BaseModel):
    """Internal inference task representation."""

    # Identity
    id: str = Field(..., description="Unique task ID")

    # Model and LoRA
    model: str = Field(..., description="Model identifier")
    lora_path_name: str = Field(..., description="LoRA directory name (job name)")
    lora_paths: List[Union[str, Dict[str, str]]] = Field(
        ..., description="Full LoRA file paths (str for single LoRA, dict for MoE: {'high': path, 'low': path})"
    )

    # Status
    status: RequestStatus = Field(default=RequestStatus.QUEUED)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # Inputs (request parameters)
    inputs: Dict[str, Any] = Field(default_factory=dict)

    # Outputs (inference results)
    outputs: Optional[Dict[str, Any]] = None

    # Error tracking
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def mark_as_processing(self):
        """Mark task as processing."""
        self.status = RequestStatus.PROCESSING
        self.started_at = datetime.now(timezone.utc)

    def mark_as_succeeded(self, outputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Mark task as succeeded."""
        self.status = RequestStatus.SUCCEEDED
        self.finished_at = datetime.now(timezone.utc)
        self.outputs = outputs
        if metadata:
            self.metadata.update(metadata)

    def mark_as_failed(self, error: str, details: Optional[Dict[str, Any]] = None):
        """Mark task as failed."""
        self.status = RequestStatus.FAILED
        self.finished_at = datetime.now(timezone.utc)
        self.error = error
        self.error_details = details

    def mark_as_cancelled(self):
        """Mark task as cancelled."""
        self.status = RequestStatus.CANCELLED
        self.finished_at = datetime.now(timezone.utc)

    @property
    def inference_time(self) -> Optional[float]:
        """Calculate inference time in seconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None
