"""
Tests for in-memory storage.
"""

import pytest
from datetime import datetime

from src.libs.storage import InMemoryStorage
from src.schemas.task import InferenceTask, RequestStatus


@pytest.fixture
def storage():
    """Create a storage instance for testing."""
    return InMemoryStorage(max_size=10)


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return InferenceTask(
        id="test-task-1",
        model="flux",
        lora_path_name="my_lora",
        lora_paths=["/path/to/lora.safetensors"],
    )


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    def test_create_and_get(self, storage, sample_task):
        """create and get should work correctly."""
        storage.create(sample_task)
        retrieved = storage.get(sample_task.id)
        assert retrieved is not None
        assert retrieved.id == sample_task.id
        assert retrieved.model == sample_task.model

    def test_get_nonexistent(self, storage):
        """get should return None for nonexistent ID."""
        assert storage.get("nonexistent") is None

    def test_update(self, storage, sample_task):
        """update should modify existing task."""
        storage.create(sample_task)

        sample_task.mark_as_processing()
        storage.update(sample_task)

        retrieved = storage.get(sample_task.id)
        assert retrieved.status == RequestStatus.PROCESSING
        assert retrieved.started_at is not None

    def test_update_nonexistent(self, storage, sample_task):
        """update should return None for nonexistent task."""
        result = storage.update(sample_task)
        assert result is None

    def test_delete(self, storage, sample_task):
        """delete should remove task."""
        storage.create(sample_task)
        assert storage.get(sample_task.id) is not None

        result = storage.delete(sample_task.id)
        assert result is True
        assert storage.get(sample_task.id) is None

    def test_delete_nonexistent(self, storage):
        """delete should return False for nonexistent ID."""
        result = storage.delete("nonexistent")
        assert result is False

    def test_list_by_status(self, storage):
        """list_by_status should filter correctly."""
        for i in range(5):
            task = InferenceTask(
                id=f"task-{i}",
                model="flux",
                lora_path_name="my_lora",
                lora_paths=["/path/to/lora.safetensors"],
            )
            if i % 2 == 0:
                task.mark_as_processing()
            storage.create(task)

        queued = storage.list_by_status(RequestStatus.QUEUED)
        processing = storage.list_by_status(RequestStatus.PROCESSING)

        assert len(queued) == 2
        assert len(processing) == 3

    def test_count(self, storage):
        """count should return correct number."""
        assert storage.count() == 0

        for i in range(5):
            task = InferenceTask(
                id=f"task-{i}",
                model="flux",
                lora_path_name="my_lora",
                lora_paths=["/path/to/lora.safetensors"],
            )
            storage.create(task)

        assert storage.count() == 5

    def test_clear(self, storage):
        """clear should remove all tasks."""
        for i in range(5):
            task = InferenceTask(
                id=f"task-{i}",
                model="flux",
                lora_path_name="my_lora",
                lora_paths=["/path/to/lora.safetensors"],
            )
            storage.create(task)

        assert storage.count() == 5
        storage.clear()
        assert storage.count() == 0

    def test_cleanup_when_full(self, storage):
        """Storage should cleanup old completed tasks when full."""
        # Fill storage with completed tasks
        for i in range(10):
            task = InferenceTask(
                id=f"task-{i}",
                model="flux",
                lora_path_name="my_lora",
                lora_paths=["/path/to/lora.safetensors"],
            )
            task.mark_as_succeeded({"images": []})
            storage.create(task)

        # Add one more task
        new_task = InferenceTask(
            id="new-task",
            model="flux",
            lora_path_name="my_lora",
            lora_paths=["/path/to/lora.safetensors"],
        )
        storage.create(new_task)

        # Should have cleaned up some old tasks
        assert storage.count() <= 10
        assert storage.get("new-task") is not None
