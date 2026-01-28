"""
Tests for model types.
"""

import pytest

from src.schemas.models import ModelType, get_supported_models


class TestModelType:
    """Tests for ModelType enum."""

    def test_all_models_have_value(self):
        """All model types should have a string value."""
        for model in ModelType:
            assert isinstance(model.value, str)
            assert len(model.value) > 0

    def test_model_values_are_unique(self):
        """Model type values should be unique."""
        values = [m.value for m in ModelType]
        assert len(values) == len(set(values))

    def test_get_supported_models(self):
        """get_supported_models should return all model values."""
        models = get_supported_models()
        assert len(models) == len(ModelType)
        for model_type in ModelType:
            assert model_type.value in models

    def test_expected_models_exist(self):
        """Expected models should exist in enum."""
        expected = [
            "flux",
            "flux_kontext",
            "flux2",
            "flux2_diffusers",
            "flex1",
            "flex2",
            "sd15",
            "sdxl",
            "qwen_image",
            "qwen_image_edit",
            "zimage_turbo",
            "zimage_deturbo",
            "wan21_14b",
            "wan21_1b",
            "wan22_5b",
            "wan22_14b_t2v",
            "wan22_14b_i2v",
            "chroma",
            "hidream",
            "hidream_e1",
            "lumina2",
            "omnigen2",
        ]
        models = get_supported_models()
        for model in expected:
            assert model in models, f"Missing model: {model}"
