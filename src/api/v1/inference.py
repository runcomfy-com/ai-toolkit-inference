"""
Inference API routes.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Union

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from dependency_injector.wiring import Provide, inject

from ...containers import Container
from ...config import Settings
from ...libs.storage import InMemoryStorage
from ...schemas.request import InferenceInput, LoraItem
from ...schemas.response import (
    InferenceResponse,
    InferenceStatusResponse,
    InferenceResultResponse,
)
from ...schemas.task import InferenceTask, RequestStatus
from ...schemas.models import ModelType, get_supported_models
from ...pipelines import get_pipeline_config, PIPELINE_REGISTRY
from ...tasks.executor import InferenceExecutor
from ...services.pipeline_manager import is_url
from ...libs.log_context import set_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["inference"])

# Models that use MoE LoRA format (high_noise, low_noise)
MOE_LORA_MODELS = {"wan22_14b_t2v", "wan22_14b_i2v"}


def _get_lora_paths(workflows_base_path: str, loras: list) -> list:
    """
    Get LoRA file paths from LoRA items.

    Supports three formats:
    1. Local files (path): ["file1.safetensors", "nested/file2.safetensors"]
    2. URLs: ["https://example.com/lora.safetensors"]
    3. MoE format: [{"low": "xxx", "high": "xxx"}] (values can be local files or URLs)

    Path structure for local files: {workflows_base_path}/{path}
    URLs are passed through as-is.
    """
    if not loras:
        return []

    # Check if it's MoE format (list of dicts)
    first_item = loras[0]
    if isinstance(first_item, dict):
        # MoE format: return list of dicts with resolved paths
        return _get_moe_lora_paths(workflows_base_path, loras)

    # Standard format: list of LoraItem (can be local files or URLs)
    lora_paths = []
    for lora in loras:
        lora_path_value = lora.path if isinstance(lora, LoraItem) else lora
        # If it's a URL, pass through as-is
        if is_url(lora_path_value):
            lora_paths.append(lora_path_value)
        else:
            # Local file: construct full path
            lora_path = os.path.join(workflows_base_path, lora_path_value)
            lora_paths.append(lora_path)
    return lora_paths


def _get_moe_lora_paths(workflows_base_path: str, loras: list) -> list:
    """
    Get LoRA paths for MoE format.

    Returns list of dicts with resolved paths: [{"low": "/path/to/low.safetensors", "high": "/path/to/high.safetensors"}]
    URLs are passed through as-is.
    """
    resolved = {}
    for config in loras:
        if isinstance(config, LoraItem):
            if not config.transformer:
                continue
            value = config.path
            if is_url(value):
                resolved[config.transformer] = value
            else:
                resolved[config.transformer] = os.path.join(workflows_base_path, value)
        else:
            for key, value in config.items():
                if not value:
                    continue
                if is_url(value):
                    resolved[key] = value
                else:
                    resolved[key] = os.path.join(workflows_base_path, value)
    return [resolved] if resolved else []


def _get_lora_scale(loras: List[LoraItem]) -> Union[float, None]:
    """Get a single LoRA scale from LoRA items (if provided)."""
    scales = [lora.network_multiplier for lora in loras if lora.network_multiplier is not None]
    if not scales:
        return None
    if len({s for s in scales}) > 1:
        logger.warning("Multiple LoRA scales provided; using the first value.")
    return scales[0]


def _get_lora_scales(loras: List[LoraItem]) -> Union[float, Dict[str, float], None]:
    """Get LoRA scale(s) from LoRA items (float for non-MoE, dict for MoE)."""
    if not loras:
        return None
    has_transformer = any(lora.transformer for lora in loras)
    if not has_transformer:
        return _get_lora_scale(loras)
    scales: Dict[str, float] = {}
    for lora in loras:
        if lora.transformer:
            scales[lora.transformer] = lora.network_multiplier
    return scales


def _validate_moe_lora_files(lora_paths: list) -> List[str]:
    """
    Validate MoE LoRA files exist (skips URL validation).

    Returns list of missing file paths.
    """
    missing = []
    for config in lora_paths:
        if config.get("low"):
            path = config["low"]
            # Skip URL validation - URLs will be downloaded at runtime
            if not is_url(path) and not os.path.exists(path):
                missing.append(path)
        if config.get("high"):
            path = config["high"]
            # Skip URL validation - URLs will be downloaded at runtime
            if not is_url(path) and not os.path.exists(path):
                missing.append(path)
    return missing


@router.post("/inference", response_model=InferenceResponse)
@inject
async def create_inference(
    request: InferenceInput,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(Provide[Container.config]),
    storage: InMemoryStorage = Depends(Provide[Container.storage]),
    executor: InferenceExecutor = Depends(Provide[Container.inference_executor]),
):
    """
    Submit an inference request.

    The request will be queued and processed asynchronously.
    Use the returned status_url and result_url to check progress and get results.
    """
    # Validate model
    request_input = request
    pipeline_config = get_pipeline_config(request_input.model)
    if not pipeline_config:
        supported_models = get_supported_models()
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "MODEL_NOT_FOUND",
                    "message": f"Model '{request_input.model}' is not supported",
                    "details": {"supported_models": supported_models},
                }
            },
        )

    # Validate loras based on model

    if len(request_input.loras) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "LORA_FILE_REQUIRED",
                    "message": "At least one LoRA is required",
                    "details": {"loras": request_input.loras},
                }
            },
        )

    # Check if MoE format (transformer field provided)
    transformers = [lora.transformer for lora in request_input.loras]
    has_transformer = any(t is not None for t in transformers)
    missing_transformer = any(t is None for t in transformers)

    if request_input.model in MOE_LORA_MODELS:
        # MoE models: require transformer field and only low/high once
        if missing_transformer:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "MOE_FORMAT_REQUIRED",
                        "message": f"Model '{request_input.model}' requires MoE LoRA format with transformer=low/high",
                        "details": {"loras": request_input.loras},
                    }
                },
            )
        if len(request_input.loras) > 2 or len(set(transformers)) != len(transformers):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "SINGLE_MOE_CONFIG_ONLY",
                        "message": f"Model '{request_input.model}' currently only supports one low/high MoE LoRA pair",
                        "details": {"loras": request_input.loras},
                    }
                },
            )
    elif has_transformer:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "MOE_FORMAT_NOT_SUPPORTED",
                    "message": f"Model '{request_input.model}' does not support MoE LoRA format",
                    "details": {"loras": request_input.loras},
                }
            },
        )
    elif len(request_input.loras) > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "SINGLE_LORA_ONLY",
                    "message": f"Model '{request_input.model}' only supports 1 LoRA file",
                    "details": {"loras": request_input.loras},
                }
            },
        )

    # Get paths (can be empty if loras is empty)
    if request_input.model in MOE_LORA_MODELS:
        lora_paths = _get_moe_lora_paths(settings.workflows_base_path, request_input.loras)
    else:
        lora_paths = _get_lora_paths(settings.workflows_base_path, request_input.loras)

    # Validate LoRA files exist
    if request_input.model in MOE_LORA_MODELS:
        missing_files = _validate_moe_lora_files(lora_paths)
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "LORA_NOT_FOUND",
                        "message": f"LoRA file(s) not found: {missing_files}",
                        "details": {"missing_files": missing_files},
                    }
                },
            )
    else:
        for lora_path in lora_paths:
            # Skip URL validation - URLs will be downloaded at runtime
            if not is_url(lora_path) and not os.path.exists(lora_path):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "code": "LORA_NOT_FOUND",
                            "message": f"LoRA file not found: {lora_path}",
                            "details": {"lora_path": lora_path},
                        }
                    },
                )

    # Validate control image for models that require it
    prompts = request.get_prompts()
    if pipeline_config.requires_control_image:
        for i, prompt_item in enumerate(prompts):
            # Check if any control image is provided
            if not prompt_item.get_control_image() and not prompt_item.get_control_images():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "code": "CONTROL_IMAGE_REQUIRED",
                            "message": f"Model '{request_input.model}' requires a control image for prompt {i}",
                            "details": {"model": request_input.model, "prompt_index": i},
                        }
                    },
                )

    # Create request ID and set it for logging context
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    now = datetime.now(timezone.utc)

    # Get hf_token: priority API request > config (from HF_TOKEN env var)
    hf_token = request_input.hf_token or settings.hf_token

    lora_scales = _get_lora_scales(request_input.loras)

    # Build inputs dict (all parameters are in prompts now)
    inputs = {
        "prompts": [p.model_dump() for p in prompts],
        "hf_token": hf_token,
        "trigger_word": request_input.trigger_word,
        "lora_scales": lora_scales,
    }

    # Create task
    task = InferenceTask(
        id=request_id,
        model=request_input.model,
        lora_path_name="",
        lora_paths=lora_paths,
        status=RequestStatus.QUEUED,
        created_at=now,
        inputs=inputs,
    )

    # Store task
    storage.create(task)

    # Add background task to process inference
    background_tasks.add_task(executor.execute, task)

    # Build response URLs
    base_url = settings.base_url.rstrip("/")
    status_url = f"{base_url}/v1/requests/{request_id}/status"
    result_url = f"{base_url}/v1/requests/{request_id}/result"

    logger.info(f"Created inference request: model={request_input.model}, lora_paths={lora_paths}")

    return InferenceResponse(
        request_id=request_id,
        status="queued",
        status_url=status_url,
        result_url=result_url,
        created_at=now.isoformat() + "Z",
    )


@router.get("/requests/{request_id}/status", response_model=InferenceStatusResponse)
@inject
async def get_status(
    request_id: str,
    storage: InMemoryStorage = Depends(Provide[Container.storage]),
):
    """
    Get the status of an inference request.
    """
    task = storage.get(request_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "REQUEST_NOT_FOUND",
                    "message": f"Request not found: {request_id}",
                    "details": {"request_id": request_id},
                }
            },
        )

    return InferenceStatusResponse(
        request_id=task.id,
        status=task.status.to_api_status(),
        created_at=task.created_at.isoformat() + "Z" if task.created_at else None,
        started_at=task.started_at.isoformat() + "Z" if task.started_at else None,
        finished_at=task.finished_at.isoformat() + "Z" if task.finished_at else None,
        error=task.error if task.status == RequestStatus.FAILED else None,
    )


@router.get("/requests/{request_id}/result", response_model=InferenceResultResponse)
@inject
async def get_result(
    request_id: str,
    storage: InMemoryStorage = Depends(Provide[Container.storage]),
):
    """
    Get the result of an inference request.
    """
    task = storage.get(request_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "REQUEST_NOT_FOUND",
                    "message": f"Request not found: {request_id}",
                    "details": {"request_id": request_id},
                }
            },
        )

    return InferenceResultResponse(
        request_id=task.id,
        status=task.status.to_api_status(),
        outputs=task.outputs,
        error=task.error if task.status == RequestStatus.FAILED else None,
        created_at=task.created_at.isoformat() + "Z" if task.created_at else None,
        finished_at=task.finished_at.isoformat() + "Z" if task.finished_at else None,
        metadata=task.metadata if task.metadata else None,
    )


@router.get("/models")
async def list_models():
    """
    List all supported models with their configurations.
    """
    models = []
    for model_type, pipeline_class in PIPELINE_REGISTRY.items():
        config = pipeline_class.CONFIG
        models.append(
            {
                "id": model_type.value,
                "base_model": config.base_model,
                "resolution_divisor": config.resolution_divisor,
                "default_steps": config.default_steps,
                "default_guidance_scale": config.default_guidance_scale,
                "requires_control_image": config.requires_control_image,
                "is_video_model": config.is_video_model,
                "default_num_frames": config.default_num_frames if config.is_video_model else None,
                "default_fps": config.default_fps if config.is_video_model else None,
            }
        )

    return {"models": models}
