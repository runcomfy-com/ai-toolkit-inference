"""
Unit tests for offload_mode functionality.

These tests verify:
1. BasePipeline correctly handles offload_mode initialization
2. Pipeline cache key correctly includes offload_mode
3. Offload modes don't stack (only one is applied)
"""

import pytest
from unittest.mock import MagicMock, patch


class TestBasePipelineOffloadMode:
    """Tests for BasePipeline offload_mode handling."""

    def test_offload_mode_default_is_model(self):
        """Test that default offload_mode is 'model'."""
        from src.pipelines.base import BasePipeline

        # Create a minimal concrete subclass for testing
        class _TestPipeline(BasePipeline):
            from src.schemas.models import ModelType
            from src.pipelines.base import PipelineConfig

            CONFIG = PipelineConfig(
                model_type=ModelType.SD15,
                base_model="test",
                resolution_divisor=8,
                default_steps=20,
                default_guidance_scale=7.0,
            )

            def _load_pipeline(self):
                self.pipe = MagicMock()

            def _run_inference(self, **kwargs):
                return {"image": MagicMock()}

        pipe = _TestPipeline(device="cpu")
        assert pipe.offload_mode == "model"

    def test_offload_mode_none(self):
        """Test offload_mode='none'."""
        from src.pipelines.base import BasePipeline

        class _TestPipeline(BasePipeline):
            from src.schemas.models import ModelType
            from src.pipelines.base import PipelineConfig

            CONFIG = PipelineConfig(
                model_type=ModelType.SD15,
                base_model="test",
                resolution_divisor=8,
                default_steps=20,
                default_guidance_scale=7.0,
            )

            def _load_pipeline(self):
                self.pipe = MagicMock()

            def _run_inference(self, **kwargs):
                return {"image": MagicMock()}

        pipe = _TestPipeline(device="cpu", offload_mode="none")
        assert pipe.offload_mode == "none"

    def test_offload_mode_sequential(self):
        """Test offload_mode='sequential'."""
        from src.pipelines.base import BasePipeline

        class _TestPipeline(BasePipeline):
            from src.schemas.models import ModelType
            from src.pipelines.base import PipelineConfig

            CONFIG = PipelineConfig(
                model_type=ModelType.SD15,
                base_model="test",
                resolution_divisor=8,
                default_steps=20,
                default_guidance_scale=7.0,
            )

            def _load_pipeline(self):
                self.pipe = MagicMock()

            def _run_inference(self, **kwargs):
                return {"image": MagicMock()}

        pipe = _TestPipeline(device="cpu", offload_mode="sequential")
        assert pipe.offload_mode == "sequential"

    def test_apply_offload_mode_calls_correct_method(self):
        """Test that _apply_offload_mode calls the correct offload method."""
        from src.pipelines.base import BasePipeline

        class _TestPipeline(BasePipeline):
            from src.schemas.models import ModelType
            from src.pipelines.base import PipelineConfig

            CONFIG = PipelineConfig(
                model_type=ModelType.SD15,
                base_model="test",
                resolution_divisor=8,
                default_steps=20,
                default_guidance_scale=7.0,
            )

            def _load_pipeline(self):
                self.pipe = MagicMock()
                self.pipe.enable_model_cpu_offload = MagicMock()
                self.pipe.enable_sequential_cpu_offload = MagicMock()
                self.pipe.to = MagicMock()

            def _run_inference(self, **kwargs):
                return {"image": MagicMock()}

        # Test model offload
        pipe_model = _TestPipeline(device="cuda", offload_mode="model")
        pipe_model._load_pipeline()
        pipe_model._apply_offload_mode()
        pipe_model.pipe.enable_model_cpu_offload.assert_called_once()
        pipe_model.pipe.enable_sequential_cpu_offload.assert_not_called()
        pipe_model.pipe.to.assert_not_called()

        # Test sequential offload
        pipe_seq = _TestPipeline(device="cuda", offload_mode="sequential")
        pipe_seq._load_pipeline()
        pipe_seq._apply_offload_mode()
        pipe_seq.pipe.enable_sequential_cpu_offload.assert_called_once()

        # Test no offload
        pipe_none = _TestPipeline(device="cuda", offload_mode="none")
        pipe_none._load_pipeline()
        pipe_none._apply_offload_mode()
        pipe_none.pipe.to.assert_called_once_with("cuda")


class TestPipelineCacheKeyOffloadMode:
    """Tests for pipeline cache key including offload_mode."""

    def test_cache_key_includes_offload_mode(self):
        """Test that PipelineCacheKey includes offload_mode."""
        from comfyui_nodes.rc_common import PipelineCacheKey

        key1 = PipelineCacheKey(
            model_id="test",
            pipeline_id="test:TestPipeline",
            offload_mode="model",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )
        key2 = PipelineCacheKey(
            model_id="test",
            pipeline_id="test:TestPipeline",
            offload_mode="sequential",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )
        key3 = PipelineCacheKey(
            model_id="test",
            pipeline_id="test:TestPipeline",
            offload_mode="none",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )

        # Different offload modes should produce different keys
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # Same offload mode should produce equal keys
        key1_copy = PipelineCacheKey(
            model_id="test",
            pipeline_id="test:TestPipeline",
            offload_mode="model",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )
        assert key1 == key1_copy

    def test_get_or_load_pipeline_different_offload_modes_cause_reload(self):
        """Test that changing offload_mode causes pipeline reload."""
        from comfyui_nodes.rc_common import get_or_load_pipeline, _PIPELINE_CACHE, PipelineCacheKey

        # Clear cache
        _PIPELINE_CACHE["instance"] = None
        _PIPELINE_CACHE["key"] = None

        # Create mock pipeline constructor
        mock_instances = []

        def mock_ctor(*args, **kwargs):
            instance = MagicMock()
            instance.load = MagicMock()
            instance.unload = MagicMock()
            mock_instances.append(instance)
            return instance

        # Load with offload_mode="model"
        pipe1 = get_or_load_pipeline(
            model_id="test",
            pipeline_ctor=mock_ctor,
            offload_mode="model",
            hf_token=None,
            lora_paths=[],
            lora_scale=1.0,
        )

        assert len(mock_instances) == 1
        assert _PIPELINE_CACHE["key"].offload_mode == "model"

        # Load with same offload_mode should return cached instance
        pipe2 = get_or_load_pipeline(
            model_id="test",
            pipeline_ctor=mock_ctor,
            offload_mode="model",
            hf_token=None,
            lora_paths=[],
            lora_scale=1.0,
        )

        assert len(mock_instances) == 1  # No new instance created
        assert pipe1 is pipe2

        # Load with different offload_mode should create new instance
        pipe3 = get_or_load_pipeline(
            model_id="test",
            pipeline_ctor=mock_ctor,
            offload_mode="sequential",
            hf_token=None,
            lora_paths=[],
            lora_scale=1.0,
        )

        assert len(mock_instances) == 2  # New instance created
        assert pipe3 is not pipe1
        assert _PIPELINE_CACHE["key"].offload_mode == "sequential"

        # Cleanup
        _PIPELINE_CACHE["instance"] = None
        _PIPELINE_CACHE["key"] = None


class TestOffloadModeNoStacking:
    """Tests to verify that offload modes don't stack (only one is applied)."""

    def test_sequential_does_not_also_call_model_offload(self):
        """Test that sequential offload doesn't also enable model offload."""
        from src.pipelines.base import BasePipeline

        class _TestPipeline(BasePipeline):
            from src.schemas.models import ModelType
            from src.pipelines.base import PipelineConfig

            CONFIG = PipelineConfig(
                model_type=ModelType.SD15,
                base_model="test",
                resolution_divisor=8,
                default_steps=20,
                default_guidance_scale=7.0,
            )

            def _load_pipeline(self):
                self.pipe = MagicMock()
                self.pipe.enable_model_cpu_offload = MagicMock()
                self.pipe.enable_sequential_cpu_offload = MagicMock()
                self.pipe.to = MagicMock()

            def _run_inference(self, **kwargs):
                return {"image": MagicMock()}

        pipe = _TestPipeline(device="cuda", offload_mode="sequential")
        pipe._load_pipeline()
        pipe._apply_offload_mode()

        # Sequential should be called
        pipe.pipe.enable_sequential_cpu_offload.assert_called_once()
        # Model offload should NOT be called
        pipe.pipe.enable_model_cpu_offload.assert_not_called()
        # .to() should NOT be called
        pipe.pipe.to.assert_not_called()

    def test_model_offload_does_not_call_sequential(self):
        """Test that model offload doesn't also enable sequential offload."""
        from src.pipelines.base import BasePipeline

        class _TestPipeline(BasePipeline):
            from src.schemas.models import ModelType
            from src.pipelines.base import PipelineConfig

            CONFIG = PipelineConfig(
                model_type=ModelType.SD15,
                base_model="test",
                resolution_divisor=8,
                default_steps=20,
                default_guidance_scale=7.0,
            )

            def _load_pipeline(self):
                self.pipe = MagicMock()
                self.pipe.enable_model_cpu_offload = MagicMock()
                self.pipe.enable_sequential_cpu_offload = MagicMock()
                self.pipe.to = MagicMock()

            def _run_inference(self, **kwargs):
                return {"image": MagicMock()}

        pipe = _TestPipeline(device="cuda", offload_mode="model")
        pipe._load_pipeline()
        pipe._apply_offload_mode()

        # Model offload should be called
        pipe.pipe.enable_model_cpu_offload.assert_called_once()
        # Sequential should NOT be called
        pipe.pipe.enable_sequential_cpu_offload.assert_not_called()
        # .to() should NOT be called
        pipe.pipe.to.assert_not_called()
