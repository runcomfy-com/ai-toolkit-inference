"""
Optional ComfyUI integration helpers.

This module is safe to import outside ComfyUI:
- It does NOT import `comfy.*` at import time.
- It only tries to import ComfyUI lazily when you request an observer.

Design:
- Core pipelines remain Comfy-agnostic.
- ComfyUI node wrappers can wrap `pipe.generate(...)` calls in the provided
  context manager to enable per-step progress updates + interrupt handling.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple, Type

from .base import PipelineStepObserver, pipeline_step_observer


def _try_import_comfy() -> Tuple[Optional[Type], Optional[Callable[[], None]]]:
    """Return (ProgressBar, throw_exception_if_processing_interrupted) if available."""
    try:
        import comfy.model_management as model_management  # type: ignore
        import comfy.utils as comfy_utils  # type: ignore

        return comfy_utils.ProgressBar, model_management.throw_exception_if_processing_interrupted
    except Exception:
        return None, None


def make_comfy_step_observer(total_steps: int) -> Optional[PipelineStepObserver]:
    """Create a per-step observer backed by ComfyUI progress + interrupt APIs."""
    ProgressBar, throw_interrupt = _try_import_comfy()
    if ProgressBar is None or throw_interrupt is None:
        return None

    total_steps_i = int(total_steps)
    pbar = ProgressBar(total_steps_i)

    def _observer(step_index: int, total: int, timestep) -> None:
        # Allow ComfyUI's cancel button to interrupt generation.
        throw_interrupt()
        # Update Comfy progress bar (expects absolute progress).
        try:
            value = int(step_index) + 1
        except Exception:
            value = 1
        pbar.update_absolute(value, int(total))

    return _observer


@contextmanager
def comfy_pipeline_observer(total_steps: int) -> Iterator[None]:
    """Context manager that enables ComfyUI progress + interrupt for one run."""
    observer = make_comfy_step_observer(total_steps)
    if observer is None:
        yield
        return

    with pipeline_step_observer(observer):
        yield

