"""
Integration tests for pipeline latent functionality.

These tests require a GPU and will be skipped if CUDA is not available.
Run with: pytest tests/test_pipeline_latent_integration.py -v --run-integration
"""

import pytest
import torch
import os


# Skip all tests if not running integration tests
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() != "true"
    and "--run-integration" not in os.environ.get("PYTEST_ADDOPTS", ""),
    reason="Integration tests skipped. Set RUN_INTEGRATION_TESTS=true to run.",
)


def pytest_configure(config):
    """Add custom marker for integration tests."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires GPU)"
    )


class TestSDXLLatentIntegration:
    """Integration tests for SDXL latent workflow."""

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip if CUDA not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture
    def sdxl_pipeline(self):
        """Load SDXL pipeline with CPU offload."""
        from src.pipelines.sdxl import SDXLPipeline

        pipe = SDXLPipeline(device="cuda", offload_mode="model")
        pipe.load(lora_paths=[], lora_scale=1.0)
        yield pipe
        try:
            pipe.unload()
        except Exception:
            pass
        torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_sdxl_generate_latent_output(self, sdxl_pipeline):
        """Test that SDXL can generate latent output."""
        result = sdxl_pipeline.generate(
            prompt="a simple test image",
            width=256,
            height=256,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
        )

        assert "latents" in result
        latents = result["latents"]

        # Check latent shape (256/8 = 32)
        assert latents.shape == (1, 4, 32, 32)
        assert latents.dtype in (torch.float32, torch.float16, torch.bfloat16)

    @pytest.mark.integration
    def test_sdxl_decode_latent(self, sdxl_pipeline):
        """Test that SDXL can decode latents to images."""
        # First generate latent
        result = sdxl_pipeline.generate(
            prompt="a simple test image",
            width=256,
            height=256,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
        )
        latents = result["latents"]

        # Decode latent
        image = sdxl_pipeline.decode_latent_to_image(latents)

        assert image.size == (256, 256)
        assert image.mode == "RGB"

    @pytest.mark.integration
    def test_sdxl_latent_upscale_refine(self, sdxl_pipeline):
        """Test full latent upscale + refine workflow."""
        import torch.nn.functional as F

        # Step 1: Generate at small size
        result1 = sdxl_pipeline.generate(
            prompt="a beautiful landscape",
            width=256,
            height=256,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
        )
        latents1 = result1["latents"]
        assert latents1.shape == (1, 4, 32, 32)

        # Step 2: Upscale latent 2x
        upscaled = F.interpolate(
            latents1.float(), scale_factor=2.0, mode="nearest"
        ).to(latents1.dtype)
        assert upscaled.shape == (1, 4, 64, 64)

        # Step 3: Refine with denoise
        result2 = sdxl_pipeline.generate(
            prompt="a beautiful landscape",
            width=512,
            height=512,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
            latents=upscaled,
            denoise_strength=0.5,
        )
        latents2 = result2["latents"]
        assert latents2.shape == (1, 4, 64, 64)

        # Step 4: Decode final image
        image = sdxl_pipeline.decode_latent_to_image(latents2)
        assert image.size == (512, 512)

    @pytest.mark.integration
    def test_sdxl_backward_compatible_generate(self, sdxl_pipeline):
        """Test that old-style generate() calls still work."""
        # Old-style call without output_type, latents, or denoise_strength
        result = sdxl_pipeline.generate(
            prompt="a simple test image",
            width=256,
            height=256,
            num_inference_steps=5,
            seed=42,
        )

        # Should return PIL image
        assert "image" in result
        assert result["image"].size == (256, 256)


class TestSD15LatentIntegration:
    """Integration tests for SD15 latent workflow."""

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip if CUDA not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture
    def sd15_pipeline(self):
        """Load SD15 pipeline with CPU offload."""
        from src.pipelines.sd15 import SD15Pipeline

        pipe = SD15Pipeline(device="cuda", offload_mode="model")
        pipe.load(lora_paths=[], lora_scale=1.0)
        yield pipe
        try:
            pipe.unload()
        except Exception:
            pass
        torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_sd15_generate_latent_output(self, sd15_pipeline):
        """Test that SD15 can generate latent output."""
        result = sd15_pipeline.generate(
            prompt="a simple test image",
            width=256,
            height=256,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
        )

        assert "latents" in result
        latents = result["latents"]
        assert latents.shape == (1, 4, 32, 32)

    @pytest.mark.integration
    def test_sd15_decode_latent(self, sd15_pipeline):
        """Test that SD15 can decode latents to images."""
        result = sd15_pipeline.generate(
            prompt="a simple test image",
            width=256,
            height=256,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
        )

        image = sd15_pipeline.decode_latent_to_image(result["latents"])
        assert image.size == (256, 256)


class TestQwen2512LatentIntegration:
    """Integration tests for Qwen 2512 latent workflow.
    
    This tests the full latent upscale + refine workflow that was previously
    not possible due to Qwen's packed latent format.
    """

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip if CUDA not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture
    def qwen_pipeline(self):
        """Load Qwen 2512 pipeline with CPU offload."""
        from src.pipelines.qwen_image import QwenImage2512Pipeline

        pipe = QwenImage2512Pipeline(device="cuda", offload_mode="model")
        pipe.load(lora_paths=[], lora_scale=1.0)
        yield pipe
        try:
            pipe.unload()
        except Exception:
            pass
        torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_qwen_generate_spatial_latent_output(self, qwen_pipeline):
        """Test that Qwen returns spatial latents [B,C,H/8,W/8] not packed [B,N,D]."""
        result = qwen_pipeline.generate(
            prompt="a simple test image",
            width=512,
            height=512,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
        )

        assert "latents" in result
        latents = result["latents"]

        # Qwen should return spatial latents: 512/8 = 64, with 16 channels
        assert latents.ndim == 4, f"Expected 4D spatial latent, got {latents.ndim}D"
        assert latents.shape == (1, 16, 64, 64), f"Unexpected shape: {latents.shape}"

    @pytest.mark.integration
    def test_qwen_decode_latent(self, qwen_pipeline):
        """Test that Qwen can decode spatial latents to images."""
        result = qwen_pipeline.generate(
            prompt="a simple test image",
            width=512,
            height=512,
            num_inference_steps=5,
            seed=42,
            output_type="latent",
        )

        image = qwen_pipeline.decode_latent_to_image(result["latents"])
        assert image.size == (512, 512)
        assert image.mode == "RGB"

    @pytest.mark.integration
    def test_qwen_latent_upscale_refine_full_workflow(self, qwen_pipeline):
        """Test full Qwen latent upscale + refine workflow.
        
        This is the key workflow that was not possible before:
        1. Generate at lower resolution with output_type="latent"
        2. Upscale the latent using bilinear interpolation
        3. Refine with denoise_strength < 1.0 using img2img pipeline
        4. Decode to final high-resolution image
        """
        import torch.nn.functional as F

        # Step 1: Generate at 512x512 with latent output
        result1 = qwen_pipeline.generate(
            prompt="a majestic lion in the savanna, golden hour lighting",
            negative_prompt="blurry, low quality",
            width=512,
            height=512,
            num_inference_steps=10,
            guidance_scale=4.0,
            seed=42,
            output_type="latent",
        )
        latents1 = result1["latents"]
        
        # Verify spatial format
        assert latents1.ndim == 4
        assert latents1.shape == (1, 16, 64, 64)  # 512/8 = 64

        # Step 2: Upscale latent 1.5x using bilinear interpolation
        upscaled = F.interpolate(
            latents1, scale_factor=1.5, mode="bilinear", align_corners=False
        )
        assert upscaled.shape == (1, 16, 96, 96)  # 64 * 1.5 = 96

        # Calculate target image size (must be divisible by 16 for Qwen)
        vae_sf = qwen_pipeline._vae_scale_factor()
        new_height = int(upscaled.shape[2] * vae_sf)  # 96 * 8 = 768
        new_width = int(upscaled.shape[3] * vae_sf)
        new_height = (new_height // 16) * 16
        new_width = (new_width // 16) * 16
        assert new_height == 768
        assert new_width == 768

        # Step 3: Refine with denoise_strength=0.35 using img2img
        assert qwen_pipeline._img2img_pipe is not None, "img2img pipeline should be available"
        
        result2 = qwen_pipeline.generate(
            prompt="a majestic lion in the savanna, golden hour lighting, highly detailed",
            negative_prompt="blurry, low quality",
            width=new_width,
            height=new_height,
            num_inference_steps=10,
            guidance_scale=4.0,
            seed=42,
            output_type="latent",
            latents=upscaled,
            denoise_strength=0.35,
        )
        refined_latent = result2["latents"]
        
        # Refined latent should match target size
        assert refined_latent.shape == (1, 16, 96, 96)

        # Step 4: Decode to final image
        image = qwen_pipeline.decode_latent_to_image(refined_latent)
        assert image.size == (768, 768)
        assert image.mode == "RGB"

    @pytest.mark.integration
    def test_qwen_backward_compatible_pil_output(self, qwen_pipeline):
        """Test that old-style generate() with PIL output still works."""
        result = qwen_pipeline.generate(
            prompt="a simple test image",
            width=512,
            height=512,
            num_inference_steps=5,
            seed=42,
            # No output_type specified - should default to "pil"
        )

        assert "image" in result
        assert result["image"].size == (512, 512)
        assert result["image"].mode == "RGB"

    @pytest.mark.integration  
    def test_qwen_img2img_pipeline_shares_components(self, qwen_pipeline):
        """Test that img2img pipeline shares VAE/transformer to save VRAM."""
        # Both pipelines should share the same VAE instance
        assert qwen_pipeline._img2img_pipe is not None
        assert qwen_pipeline._img2img_pipe.vae is qwen_pipeline.pipe.vae
        assert qwen_pipeline._img2img_pipe.transformer is qwen_pipeline.pipe.transformer


class TestPipelineCacheIntegration:
    """Integration tests for pipeline cache with different pipeline types."""

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip if CUDA not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.mark.integration
    def test_cache_distinguishes_pipeline_types(self):
        """Test that cache correctly distinguishes different pipeline types."""
        from comfyui_nodes.rc_common import get_or_load_pipeline, _PIPELINE_CACHE
        from src.pipelines.sdxl import SDXLPipeline
        from src.pipelines.sd15 import SD15Pipeline

        # Clear cache
        _PIPELINE_CACHE["instance"] = None
        _PIPELINE_CACHE["key"] = None

        # Load SDXL
        sdxl = get_or_load_pipeline(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            pipeline_ctor=SDXLPipeline,
            offload_mode="model",
            hf_token=None,
            lora_paths=[],
            lora_scale=1.0,
        )
        assert isinstance(sdxl, SDXLPipeline)

        # Load SD15 - should replace SDXL in cache
        sd15 = get_or_load_pipeline(
            model_id="stabilityai/stable-diffusion-v1-5",
            pipeline_ctor=SD15Pipeline,
            offload_mode="model",
            hf_token=None,
            lora_paths=[],
            lora_scale=1.0,
        )
        assert isinstance(sd15, SD15Pipeline)

        # Verify cache was updated
        assert _PIPELINE_CACHE["instance"] is sd15

        # Clean up
        try:
            sd15.unload()
        except Exception:
            pass
        _PIPELINE_CACHE["instance"] = None
        _PIPELINE_CACHE["key"] = None
        torch.cuda.empty_cache()


class TestQwenUnloadIntegration:
    """Integration tests for Qwen pipeline unload behavior.
    
    These tests verify that:
    1. _img2img_pipe is properly cleared on unload
    2. Sequential offload works correctly with img2img
    """

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip if CUDA not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture
    def qwen_pipeline_for_unload(self):
        """Load Qwen 2512 pipeline for unload testing."""
        from src.pipelines.qwen_image import QwenImage2512Pipeline

        pipe = QwenImage2512Pipeline(device="cuda", offload_mode="model")
        pipe.load(lora_paths=[], lora_scale=1.0)
        yield pipe
        # Explicit unload to test the behavior
        try:
            pipe.unload()
        except Exception:
            pass
        torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_qwen_unload_clears_img2img_pipe(self, qwen_pipeline_for_unload):
        """Test that unload() properly clears _img2img_pipe."""
        pipe = qwen_pipeline_for_unload
        
        # Verify img2img pipeline was created
        assert pipe._img2img_pipe is not None, "img2img pipeline should be created after load"
        assert pipe.pipe is not None, "base pipeline should exist"
        
        # Unload
        pipe.unload()
        
        # Verify cleanup
        assert pipe._img2img_pipe is None, "_img2img_pipe should be None after unload"
        assert pipe.pipe is None, "pipe should be None after unload"
        assert getattr(pipe, "_using_sequential_offload", False) is False, "sequential offload flag should be reset"

    @pytest.mark.integration
    def test_qwen_img2img_pipe_created_after_load(self):
        """Test that _img2img_pipe is created in load(), not _load_pipeline()."""
        from src.pipelines.qwen_image import QwenImage2512Pipeline

        pipe = QwenImage2512Pipeline(device="cuda", offload_mode="model")
        
        # Before load, img2img should be None
        assert pipe._img2img_pipe is None
        
        # After load, img2img should exist
        pipe.load(lora_paths=[], lora_scale=1.0)
        assert pipe._img2img_pipe is not None
        
        pipe.unload()
        torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_qwen_refine_pass_with_cpu_offload(self, qwen_pipeline_for_unload):
        """Test that refine pass works correctly with CPU offload enabled."""
        pipe = qwen_pipeline_for_unload
        
        # First pass - generate latent
        result1 = pipe.generate(
            prompt="a test image",
            width=512,
            height=512,
            num_inference_steps=5,
            guidance_scale=4.0,
            seed=42,
            output_type="latent",
        )
        assert "latents" in result1
        latents = result1["latents"]
        
        # Second pass - refine with denoise < 1.0
        result2 = pipe.generate(
            prompt="a test image, refined",
            width=512,
            height=512,
            num_inference_steps=5,
            guidance_scale=4.0,
            seed=42,
            output_type="latent",
            latents=latents,
            denoise_strength=0.4,
        )
        
        assert "latents" in result2
        refined = result2["latents"]
        assert refined.shape == latents.shape, "Refined latent should have same shape"


class TestResolutionDivisorSnapping:
    """Tests for resolution divisor auto-snapping behavior."""

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip if CUDA not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.mark.integration
    def test_sdxl_snaps_resolution_to_divisor(self):
        """Test that SDXL snaps non-divisible resolution."""
        from src.pipelines.sdxl import SDXLPipeline
        import logging
        
        pipe = SDXLPipeline(device="cuda", offload_mode="model")
        pipe.load(lora_paths=[], lora_scale=1.0)
        
        try:
            # Request a resolution not divisible by 8 (SDXL divisor)
            # 513 should snap to 512
            result = pipe.generate(
                prompt="test",
                width=513,
                height=517,
                num_inference_steps=2,
                seed=42,
                output_type="latent",
            )
            
            latents = result["latents"]
            # Should be snapped: 512/8=64, 512/8=64
            assert latents.shape[-1] == 64, f"Width latent should be 64, got {latents.shape[-1]}"
            assert latents.shape[-2] == 64, f"Height latent should be 64, got {latents.shape[-2]}"
        finally:
            pipe.unload()
            torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_qwen_snaps_resolution_to_divisor(self):
        """Test that Qwen snaps non-divisible resolution."""
        from src.pipelines.qwen_image import QwenImage2512Pipeline
        
        pipe = QwenImage2512Pipeline(device="cuda", offload_mode="model")
        pipe.load(lora_paths=[], lora_scale=1.0)
        
        try:
            # Request a resolution not divisible by 16 (Qwen divisor)
            # 520 should snap to 512 (520 // 16 * 16 = 512)
            result = pipe.generate(
                prompt="test",
                width=520,
                height=530,
                num_inference_steps=2,
                seed=42,
                output_type="latent",
            )
            
            latents = result["latents"]
            # Should be snapped: 512/8=64, 528/8=66 (530//16*16=528)
            # Actually 530//16=33, 33*16=528, 528/8=66
            assert latents.shape[-1] == 64, f"Width latent should be 64 (512/8), got {latents.shape[-1]}"
            assert latents.shape[-2] == 66, f"Height latent should be 66 (528/8), got {latents.shape[-2]}"
        finally:
            pipe.unload()
            torch.cuda.empty_cache()
