"""
In-memory storage for inference requests.
"""

import threading
from typing import Dict, Optional, List
from datetime import datetime

from ..schemas.task import InferenceTask, RequestStatus


class InMemoryStorage:
    """Thread-safe in-memory storage for inference requests."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize storage.

        Args:
            max_size: Maximum number of requests to keep in memory.
                     Oldest completed requests are removed when limit is exceeded.
        """
        self._requests: Dict[str, InferenceTask] = {}
        self._lock = threading.RLock()
        self._max_size = max_size

    def create(self, task: InferenceTask) -> InferenceTask:
        """Create a new request in storage."""
        with self._lock:
            self._cleanup_if_needed()
            self._requests[task.id] = task
            return task

    def get(self, request_id: str) -> Optional[InferenceTask]:
        """Get a request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def update(self, task: InferenceTask) -> Optional[InferenceTask]:
        """Update an existing request."""
        with self._lock:
            if task.id in self._requests:
                self._requests[task.id] = task
                return task
            return None

    def delete(self, request_id: str) -> bool:
        """Delete a request by ID."""
        with self._lock:
            if request_id in self._requests:
                del self._requests[request_id]
                return True
            return False

    def list_by_status(self, status: RequestStatus) -> List[InferenceTask]:
        """List all requests with a specific status."""
        with self._lock:
            return [task for task in self._requests.values() if task.status == status]

    def get_next_queued(self) -> Optional[InferenceTask]:
        """Get the next queued request (FIFO order)."""
        with self._lock:
            queued = self.list_by_status(RequestStatus.QUEUED)
            if queued:
                # Sort by created_at to ensure FIFO
                queued.sort(key=lambda t: t.created_at)
                return queued[0]
            return None

    def count(self) -> int:
        """Get total number of requests in storage."""
        with self._lock:
            return len(self._requests)

    def count_by_status(self, status: RequestStatus) -> int:
        """Count requests with a specific status."""
        with self._lock:
            return len(self.list_by_status(status))

    def _cleanup_if_needed(self):
        """Remove oldest completed requests if storage is full."""
        if len(self._requests) >= self._max_size:
            # Get completed/failed/cancelled requests sorted by finished_at
            completed = [
                task
                for task in self._requests.values()
                if task.status in [RequestStatus.SUCCEEDED, RequestStatus.FAILED, RequestStatus.CANCELLED]
            ]

            if completed:
                # Sort by finished_at (oldest first)
                completed.sort(key=lambda t: t.finished_at or datetime.min)

                # Remove oldest 10% to make room
                remove_count = max(1, len(completed) // 10)
                for task in completed[:remove_count]:
                    del self._requests[task.id]

    def clear(self):
        """Clear all requests from storage."""
        with self._lock:
            self._requests.clear()
