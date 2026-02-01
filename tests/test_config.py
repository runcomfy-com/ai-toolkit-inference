"""
Tests for configuration management.
"""

import os
import pytest
from unittest.mock import patch


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Settings should have correct default values."""
        # Import fresh to get defaults
        with patch.dict(os.environ, {}, clear=True):
            from src.config import Settings

            settings = Settings()

            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.debug is False
            assert settings.device == "cuda"
            assert settings.offload_mode == "none"
            assert settings.enable_cpu_offload is False
            assert settings.workflows_base_path == "/app/ai-toolkit/lora_weights"
            assert settings.output_base_path == "/tmp/inference_output"
            assert settings.ai_toolkit_path == "/app/ai-toolkit"
            assert settings.inference_timeout == 3600

    def test_env_override(self):
        """Settings should be overridable via environment variables."""
        env_vars = {
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "DEBUG": "true",
            "DEVICE": "cpu",
            "OFFLOAD_MODE": "model",
            "WORKFLOWS_BASE_PATH": "/custom/workflows",
            "OUTPUT_BASE_PATH": "/custom/output",
            "AI_TOOLKIT_PATH": "/custom/ai-toolkit",
            "INFERENCE_TIMEOUT": "7200",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            from src.config import Settings

            settings = Settings()

            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.debug is True
            assert settings.device == "cpu"
            assert settings.offload_mode == "model"
            assert settings.workflows_base_path == "/custom/workflows"
            assert settings.output_base_path == "/custom/output"
            assert settings.ai_toolkit_path == "/custom/ai-toolkit"
            assert settings.inference_timeout == 7200

    def test_optional_fields(self):
        """Optional fields should default to None."""
        with patch.dict(os.environ, {}, clear=True):
            from src.config import Settings

            settings = Settings()

            assert settings.model_cache_dir is None
            assert settings.hf_token is None

    def test_optional_fields_with_values(self):
        """Optional fields should accept values."""
        env_vars = {
            "MODEL_CACHE_DIR": "/cache/models",
            "HF_TOKEN": "hf_test_token",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            from src.config import Settings

            settings = Settings()

            assert settings.model_cache_dir == "/cache/models"
            assert settings.hf_token == "hf_test_token"
