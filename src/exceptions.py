"""
Custom exceptions for inference server.
"""

from typing import Any, Dict, Optional


class InferenceServerError(Exception):
    """Base exception for inference server errors."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500):
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ModelNotFoundError(InferenceServerError):
    """Raised when requested model is not found or not supported."""

    def __init__(self, model: str):
        super().__init__(
            code="MODEL_NOT_FOUND",
            message=f"Model '{model}' is not supported",
            details={"model": model},
            status_code=400,
        )


class LoRANotFoundError(InferenceServerError):
    """Raised when LoRA file is not found."""

    def __init__(self, lora_path: str):
        super().__init__(
            code="LORA_NOT_FOUND",
            message=f"LoRA file not found: {lora_path}",
            details={"lora_path": lora_path},
            status_code=400,
        )


class ConfigNotFoundError(InferenceServerError):
    """Raised when config file is not found."""

    def __init__(self, config_path: str):
        super().__init__(
            code="CONFIG_NOT_FOUND",
            message=f"Config file not found: {config_path}",
            details={"config_path": config_path},
            status_code=400,
        )


class RequestNotFoundError(InferenceServerError):
    """Raised when request is not found."""

    def __init__(self, request_id: str):
        super().__init__(
            code="REQUEST_NOT_FOUND",
            message=f"Request not found: {request_id}",
            details={"request_id": request_id},
            status_code=404,
        )


class InvalidParameterError(InferenceServerError):
    """Raised when request parameters are invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(code="INVALID_PARAMETER", message=message, details=details, status_code=422)


class InferenceError(InferenceServerError):
    """Raised when inference fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(code="INFERENCE_ERROR", message=message, details=details, status_code=500)


class ControlImageRequiredError(InferenceServerError):
    """Raised when control image is required but not provided."""

    def __init__(self, model: str):
        super().__init__(
            code="CONTROL_IMAGE_REQUIRED",
            message=f"Model '{model}' requires a control image",
            details={"model": model},
            status_code=400,
        )
