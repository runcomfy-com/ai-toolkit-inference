"""
Logging context management using contextvars.

Provides request_id context for tracking logs across async/sync boundaries.
"""

import logging
from contextvars import ContextVar

# Context variable for request_id (used across async/sync contexts)
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    """Set the current request_id for logging context."""
    request_id_ctx.set(request_id)


def get_request_id() -> str:
    """Get the current request_id from logging context."""
    return request_id_ctx.get()


class RequestIdFilter(logging.Filter):
    """Logging filter that adds request_id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True
