"""
FastAPI server entry point.
"""

import json
import logging
import logging.config
import re
import time
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .containers import Container
from .api.v1 import inference_router
from .exceptions import InferenceServerError
from .libs.log_context import RequestIdFilter

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s",
        },
    },
    "filters": {
        "request_id_filter": {
            "()": RequestIdFilter,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "filters": ["request_id_filter"],
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def _sanitize_body_for_log(body: dict) -> dict:
    """Sanitize request body for logging, hiding base64 image data."""
    if not isinstance(body, dict):
        return body

    result = {}
    for k, v in body.items():
        if k == "prompts" and isinstance(v, list):
            # Sanitize each prompt
            result[k] = [_sanitize_prompt(p) for p in v]
        elif isinstance(v, str) and len(v) > 200:
            # Likely base64 data
            result[k] = f"<base64:{len(v)} chars>"
        elif isinstance(v, dict):
            result[k] = _sanitize_body_for_log(v)
        else:
            result[k] = v
    return result


def _sanitize_prompt(prompt: dict) -> dict:
    """Sanitize a single prompt dict."""
    if not isinstance(prompt, dict):
        return prompt

    result = {}
    for k, v in prompt.items():
        if k.startswith("ctrl_img") and isinstance(v, str) and len(v) > 200:
            result[k] = f"<base64:{len(v)} chars>"
        else:
            result[k] = v
    return result


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests with sanitized body."""

    # Skip logging body for these paths (health checks, etc.)
    SKIP_BODY_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        path = request.url.path

        # Log request
        log_body = method in ("POST", "PUT", "PATCH") and path not in self.SKIP_BODY_PATHS

        if log_body:
            try:
                # Read and cache body
                body_bytes = await request.body()
                if body_bytes:
                    body = json.loads(body_bytes)
                    sanitized = _sanitize_body_for_log(body)
                    # Use compact JSON format for single-line logging
                    body_str = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
                else:
                    body_str = "{}"

                logger.info(f">>> {method} {path} | Body: {body_str}")

                # Reconstruct request with cached body
                async def receive():
                    return {"type": "http.request", "body": body_bytes}

                request._receive = receive

            except Exception as e:
                logger.info(f">>> {method} {path} | Body: <failed to parse: {e}>")
        else:
            logger.info(f">>> {method} {path}")

        # Execute request
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(f"<<< {method} {path} | {response.status_code} | {duration:.3f}s")

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting inference server...")
    
    # Initialize container
    container = Container()
    app.state.container = container

    # Wire container for dependency injection
    container.wire(
        modules=[
            "src.api.v1.inference",
        ]
    )

    logger.info("Inference server started successfully")

    yield

    # Cleanup
    logger.info("Shutting down inference server...")

    # Unload any cached pipelines
    try:
        pipeline_manager = container.pipeline_manager()
        pipeline_manager.unload_all()
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

    logger.info("Inference server shut down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Inference Server",
        description="Diffusers-based inference service for AI Toolkit LoRA models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add request logging middleware (must be added before CORS)
    app.add_middleware(RequestLoggingMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    @app.exception_handler(InferenceServerError)
    async def inference_server_error_handler(request: Request, exc: InferenceServerError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Register API routers
    app.include_router(inference_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    from .config import settings

    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
