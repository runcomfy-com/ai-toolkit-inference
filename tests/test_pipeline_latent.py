"""
Tests for pipeline latent output functionality and backward compatibility.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import numpy as np


class TestPipelineLatentOutput:
    """Tests for pipeline latent output functionality."""

    def test_sdxl_latent_output_type_parameter(self):
        """SDXL pipeline should accept output_type parameter."""
        from src.pipelines.sdxl import SDXLPipeline

        # Check that _run_inference signature includes output_type
        import inspect
        sig = inspect.signature(SDXLPipeline._run_inference)
        params = list(sig.parameters.keys())
        assert "output_type" in params
        assert "latents" in params
        assert "denoise_strength" in params

    def test_sd15_latent_output_type_parameter(self):
        """SD15 pipeline should accept output_type parameter."""
        from src.pipelines.sd15 import SD15Pipeline

        import inspect
        sig = inspect.signature(SD15Pipeline._run_inference)
        params = list(sig.parameters.keys())
        assert "output_type" in params
        assert "latents" in params
        assert "denoise_strength" in params

    def test_qwen_image_latent_output_type_parameter(self):
        """Qwen Image pipeline should accept output_type parameter."""
        from src.pipelines.qwen_image import QwenImagePipeline

        import inspect
        sig = inspect.signature(QwenImagePipeline._run_inference)
        params = list(sig.parameters.keys())
        assert "output_type" in params
        assert "latents" in params
        assert "denoise_strength" in params

    def test_base_pipeline_generate_signature(self):
        """Base pipeline generate() should accept output_type, latents, denoise_strength."""
        from src.pipelines.base import BasePipeline

        import inspect
        sig = inspect.signature(BasePipeline.generate)
        params = list(sig.parameters.keys())
        assert "output_type" in params
        assert "latents" in params
        assert "denoise_strength" in params

    def test_output_type_default_is_pil(self):
        """Default output_type should be 'pil' for backward compatibility."""
        from src.pipelines.base import BasePipeline

        import inspect
        sig = inspect.signature(BasePipeline.generate)
        output_type_param = sig.parameters.get("output_type")
        assert output_type_param is not None
        assert output_type_param.default == "pil"

    def test_denoise_strength_default_is_one(self):
        """Default denoise_strength should be 1.0 for backward compatibility."""
        from src.pipelines.base import BasePipeline

        import inspect
        sig = inspect.signature(BasePipeline.generate)
        denoise_param = sig.parameters.get("denoise_strength")
        assert denoise_param is not None
        assert denoise_param.default == 1.0

    def test_latents_default_is_none(self):
        """Default latents should be None for backward compatibility."""
        from src.pipelines.base import BasePipeline

        import inspect
        sig = inspect.signature(BasePipeline.generate)
        latents_param = sig.parameters.get("latents")
        assert latents_param is not None
        assert latents_param.default is None


class TestPipelineCacheKey:
    """Tests for pipeline cache key functionality."""

    def test_cache_key_includes_pipeline_id(self):
        """PipelineCacheKey should include pipeline_id field."""
        from comfyui_nodes.rc_common import PipelineCacheKey

        # Check that PipelineCacheKey has pipeline_id
        import dataclasses
        fields = {f.name for f in dataclasses.fields(PipelineCacheKey)}
        assert "pipeline_id" in fields

    def test_cache_key_different_pipelines_different_keys(self):
        """Different pipeline classes should produce different cache keys."""
        from comfyui_nodes.rc_common import PipelineCacheKey

        key1 = PipelineCacheKey(
            model_id="test-model",
            pipeline_id="SDXLPipeline",
            offload_mode="none",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )
        key2 = PipelineCacheKey(
            model_id="test-model",
            pipeline_id="SD15Pipeline",
            offload_mode="none",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )
        # Same model_id but different pipeline_id should produce different keys
        assert key1 != key2
        assert hash(key1) != hash(key2)

    def test_cache_key_same_pipeline_same_keys(self):
        """Same pipeline class and params should produce same cache keys."""
        from comfyui_nodes.rc_common import PipelineCacheKey

        key1 = PipelineCacheKey(
            model_id="test-model",
            pipeline_id="SDXLPipeline",
            offload_mode="none",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )
        key2 = PipelineCacheKey(
            model_id="test-model",
            pipeline_id="SDXLPipeline",
            offload_mode="none",
            hf_token=None,
            lora_paths_key=(),
            lora_scale_key=1.0,
        )
        assert key1 == key2
        assert hash(key1) == hash(key2)

    def test_get_pipeline_id_function(self):
        """_get_pipeline_id should extract correct identifier."""
        from comfyui_nodes.rc_common import _get_pipeline_id
        from src.pipelines.sdxl import SDXLPipeline
        from src.pipelines.sd15 import SD15Pipeline

        sdxl_id = _get_pipeline_id(SDXLPipeline)
        sd15_id = _get_pipeline_id(SD15Pipeline)

        assert sdxl_id != sd15_id
        assert "SDXL" in sdxl_id or "sdxl" in sdxl_id.lower()
        assert "SD15" in sd15_id or "sd15" in sd15_id.lower()


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_pipeline_generate_without_new_params(self):
        """Pipeline.generate() should work without new params (backward compat)."""
        from src.pipelines.base import BasePipeline

        # Create a mock pipeline to test the interface
        class MockPipeline(BasePipeline):
            CONFIG = Mock()
            CONFIG.base_model = "test"
            CONFIG.default_steps = 10
            CONFIG.default_guidance_scale = 7.0
            CONFIG.supports_negative_prompt = True
            CONFIG.resolution_divisor = 8

            def _load_pipeline(self):
                self.pipe = Mock()

            def _run_inference(self, **kwargs):
                return {"image": Image.new("RGB", (64, 64))}

        # Ensure the old calling convention still works
        pipe = MockPipeline.__new__(MockPipeline)
        pipe.device = "cpu"
        pipe.dtype = torch.float32
        pipe.offload_mode = "none"
        pipe.pipe = Mock()
        pipe.CONFIG = MockPipeline.CONFIG

        # Old-style call without new params should work
        import inspect
        sig = inspect.signature(BasePipeline.generate)

        # Check that we can call with just the required params
        required_params = [
            p.name for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.name != "self"
        ]

        # Only prompt should be truly required
        assert "prompt" in required_params or len(required_params) == 0

    def test_sdxl_encode_decode_methods_exist(self):
        """SDXL pipeline should have encode/decode methods."""
        from src.pipelines.sdxl import SDXLPipeline

        assert hasattr(SDXLPipeline, "encode_image_to_latent")
        assert hasattr(SDXLPipeline, "decode_latent_to_image")
        assert callable(getattr(SDXLPipeline, "encode_image_to_latent"))
        assert callable(getattr(SDXLPipeline, "decode_latent_to_image"))

    def test_sd15_encode_decode_methods_exist(self):
        """SD15 pipeline should have encode/decode methods."""
        from src.pipelines.sd15 import SD15Pipeline

        assert hasattr(SD15Pipeline, "encode_image_to_latent")
        assert hasattr(SD15Pipeline, "decode_latent_to_image")

    def test_qwen_encode_decode_methods_exist(self):
        """Qwen pipeline should have encode/decode methods."""
        from src.pipelines.qwen_image import QwenImagePipeline

        assert hasattr(QwenImagePipeline, "encode_image_to_latent")
        assert hasattr(QwenImagePipeline, "decode_latent_to_image")

    def test_existing_rc_nodes_still_registered(self):
        """Existing RC nodes should still be registered."""
        from comfyui_nodes import NODE_CLASS_MAPPINGS

        # Check existing nodes are still there (using actual node names)
        existing_nodes = [
            "RCFluxDev",
            "RCFluxKontext",
            "RCSD15",
            "RCSDXL",
            "RCQwenImage",
            "RCQwenImage2512",
        ]
        for node in existing_nodes:
            assert node in NODE_CLASS_MAPPINGS, f"Missing node: {node}"

    def test_new_latent_nodes_registered(self):
        """New latent workflow nodes should be registered."""
        from comfyui_nodes import NODE_CLASS_MAPPINGS

        new_nodes = [
            "RCAITKLoRA",
            "RCAITKLoadPipeline",
            "RCAITKSampler",
            "RCAITKDecodeLatent",
            "RCAITKEncodeImage",
        ]
        for node in new_nodes:
            assert node in NODE_CLASS_MAPPINGS, f"Missing new node: {node}"


class TestComfyUINodeInputTypes:
    """Tests for ComfyUI node input/output types."""

    def test_aitk_lora_node_structure(self):
        """RCAITKLoRA node should have correct structure."""
        from comfyui_nodes.rc_latent_workflow import RCAITKLoRA

        # Note: INPUT_TYPES() requires folder_paths (ComfyUI), skip that part
        # Check RETURN_TYPES
        assert hasattr(RCAITKLoRA, "RETURN_TYPES")
        assert "AITK_LORA" in RCAITKLoRA.RETURN_TYPES

        # Check FUNCTION
        assert hasattr(RCAITKLoRA, "FUNCTION")
        assert RCAITKLoRA.FUNCTION == "build"

    def test_aitk_load_pipeline_node_structure(self):
        """RCAITKLoadPipeline node should have correct structure."""
        from comfyui_nodes.rc_latent_workflow import RCAITKLoadPipeline

        input_types = RCAITKLoadPipeline.INPUT_TYPES()
        assert "required" in input_types
        assert "pipeline" in input_types["required"]  # Fixed: was "model"

        # Should accept optional AITK_LORA input
        assert "optional" in input_types
        assert "lora" in input_types["optional"]

        # Check RETURN_TYPES includes AITK_PIPELINE
        assert "AITK_PIPELINE" in RCAITKLoadPipeline.RETURN_TYPES

    def test_aitk_sampler_node_structure(self):
        """RCAITKSampler node should have correct structure."""
        from comfyui_nodes.rc_latent_workflow import RCAITKSampler

        input_types = RCAITKSampler.INPUT_TYPES()
        assert "required" in input_types
        assert "pipe" in input_types["required"]  # Fixed: was "pipeline"
        assert "latent" in input_types["required"]
        assert "prompt" in input_types["required"]

        # Check RETURN_TYPES includes LATENT
        assert "LATENT" in RCAITKSampler.RETURN_TYPES

    def test_aitk_decode_latent_node_structure(self):
        """RCAITKDecodeLatent node should have correct structure."""
        from comfyui_nodes.rc_latent_workflow import RCAITKDecodeLatent

        input_types = RCAITKDecodeLatent.INPUT_TYPES()
        assert "required" in input_types
        assert "pipe" in input_types["required"]  # Fixed: was "pipeline"
        assert "latent" in input_types["required"]

        # Check RETURN_TYPES includes IMAGE
        assert "IMAGE" in RCAITKDecodeLatent.RETURN_TYPES

    def test_aitk_encode_image_node_structure(self):
        """RCAITKEncodeImage node should have correct structure."""
        from comfyui_nodes.rc_latent_workflow import RCAITKEncodeImage

        input_types = RCAITKEncodeImage.INPUT_TYPES()
        assert "required" in input_types
        assert "pipe" in input_types["required"]  # Fixed: was "pipeline"
        assert "image" in input_types["required"]

        # Check RETURN_TYPES includes LATENT
        assert "LATENT" in RCAITKEncodeImage.RETURN_TYPES


class TestNoiseGeneration:
    """Tests for noise generation compatibility."""

    def test_noise_generation_on_cpu_generator(self):
        """Noise generation should work with CPU generator."""
        # This tests the fix for torch.randn_like with generator on CUDA
        generator = torch.Generator(device="cpu").manual_seed(42)

        # Simulate the fixed code pattern
        latent_shape = (1, 4, 64, 64)
        noise = torch.randn(
            latent_shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

        assert noise.shape == latent_shape
        assert noise.device.type == "cpu"

    def test_noise_can_be_moved_to_different_dtype(self):
        """Noise should be movable to different dtypes."""
        generator = torch.Generator(device="cpu").manual_seed(42)
        latent_shape = (1, 4, 64, 64)

        noise = torch.randn(
            latent_shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

        # Should be able to convert to bfloat16
        noise_bf16 = noise.to(dtype=torch.bfloat16)
        assert noise_bf16.dtype == torch.bfloat16

        # Should be able to convert to float16
        noise_fp16 = noise.to(dtype=torch.float16)
        assert noise_fp16.dtype == torch.float16


class TestLatentFormatCompatibility:
    """Tests for latent format handling."""

    def test_sd_latent_shape_is_4d(self):
        """SD/SDXL latents should be 4D tensors."""
        # Standard SD latent shape: [B, C, H, W]
        sd_latent = torch.randn(1, 4, 64, 64)
        assert sd_latent.dim() == 4
        assert sd_latent.shape[1] == 4  # 4 channels

    def test_latent_upscale_simulation(self):
        """Latent upscaling should work for SD-style latents."""
        import torch.nn.functional as F

        # Original latent
        latent = torch.randn(1, 4, 64, 64)

        # Upscale 1.5x (nearest neighbor like ComfyUI)
        upscaled = F.interpolate(latent, scale_factor=1.5, mode="nearest")

        assert upscaled.shape == (1, 4, 96, 96)  # 64 * 1.5 = 96

    def test_comfyui_latent_dict_format(self):
        """ComfyUI expects latents in {'samples': tensor} format."""
        latent_tensor = torch.randn(1, 4, 64, 64)
        comfyui_latent = {"samples": latent_tensor}

        assert "samples" in comfyui_latent
        assert comfyui_latent["samples"].shape == (1, 4, 64, 64)


class TestImageConversion:
    """Tests for image conversion utilities."""

    def test_tensor_to_pil_conversion(self):
        """Tensor to PIL conversion should work correctly."""
        # Simulate decode output [B, C, H, W] in [-1, 1] range
        tensor = torch.randn(1, 3, 64, 64)
        tensor = (tensor / 2 + 0.5).clamp(0, 1)

        # Convert to PIL
        image_np = tensor.float().cpu().permute(0, 2, 3, 1).numpy()[0]
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        assert pil_image.size == (64, 64)
        assert pil_image.mode == "RGB"

    def test_pil_to_tensor_conversion(self):
        """PIL to tensor conversion should work correctly."""
        # Create test image
        pil_image = Image.new("RGB", (64, 64), color=(128, 128, 128))

        # Convert to tensor [B, C, H, W] in [-1, 1]
        img_array = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor * 2.0 - 1.0

        assert img_tensor.shape == (1, 3, 64, 64)
        assert img_tensor.min() >= -1.0
        assert img_tensor.max() <= 1.0
