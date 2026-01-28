"""
Pytest configuration and shared fixtures.
"""

import sys
import os
import pytest

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), "..")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def clean_storage():
    """Create a clean storage instance for each test."""
    from src.libs.storage import InMemoryStorage

    storage = InMemoryStorage(max_size=100)
    yield storage
    storage.clear()
