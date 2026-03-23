"""
Tests for the LTX-2.3 composable upscale workflow nodes.

Covers:
- Node registration (all 3 nodes in NODE_CLASS_MAPPINGS)
- INPUT_TYPES / RETURN_TYPES / wire type contracts
- Pipeline method signatures on LTX23Pipeline
- Example workflow JSON validity (all class_types exist in registry)
- Node execution with mocked pipeline (input→output data flow)
"""

import json
import os
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch


# ── Registration tests ──────────────────────────────────────────────


class TestNodeRegistration:
    """All 3 upscale nodes must be registered in NODE_CLASS_MAPPINGS."""

    def test_generate_latent_registered(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS
        assert "RCLTX23GenerateLatent" in NODE_CLASS_MAPPINGS

    def test_latent_upscale_registered(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS
        assert "RCLTX23LatentUpscale" in NODE_CLASS_MAPPINGS

    def test_denoise_registered(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS
        assert "RCLTX23Denoise" in NODE_CLASS_MAPPINGS

    def test_old_upscale_node_removed(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS
        assert "RCLTX23Upscale" not in NODE_CLASS_MAPPINGS

    def test_display_names(self):
        from comfyui_nodes import NODE_DISPLAY_NAME_MAPPINGS
        assert NODE_DISPLAY_NAME_MAPPINGS["RCLTX23GenerateLatent"] == "RC LTX-2.3 Generate Latent"
        assert NODE_DISPLAY_NAME_MAPPINGS["RCLTX23LatentUpscale"] == "RC LTX-2.3 Latent Upscale (2x)"
        assert NODE_DISPLAY_NAME_MAPPINGS["RCLTX23Denoise"] == "RC LTX-2.3 Denoise"


# ── Input/Output contract tests ─────────────────────────────────────


class TestNodeContracts:
    """INPUT_TYPES and RETURN_TYPES must match the LTX23_LATENT wire type."""

    def test_generate_latent_inputs(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23GenerateLatent
        inputs = RCLTX23GenerateLatent.INPUT_TYPES()
        req = inputs["required"]
        assert "pipe" in req
        assert req["pipe"] == ("AITK_PIPELINE",)
        assert "prompt" in req
        assert "width" in req
        assert "height" in req
        assert "steps" in req
        assert "cfg" in req
        assert "seed" in req

    def test_generate_latent_output_type(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23GenerateLatent
        assert RCLTX23GenerateLatent.RETURN_TYPES == ("LTX23_LATENT",)

    def test_latent_upscale_inputs(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23LatentUpscale
        inputs = RCLTX23LatentUpscale.INPUT_TYPES()
        req = inputs["required"]
        assert req["pipe"] == ("AITK_PIPELINE",)
        assert req["ltx_latent"] == ("LTX23_LATENT",)

    def test_latent_upscale_output_type(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23LatentUpscale
        assert RCLTX23LatentUpscale.RETURN_TYPES == ("LTX23_LATENT",)

    def test_denoise_inputs(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23Denoise
        inputs = RCLTX23Denoise.INPUT_TYPES()
        req = inputs["required"]
        assert req["pipe"] == ("AITK_PIPELINE",)
        assert req["ltx_latent"] == ("LTX23_LATENT",)

    def test_denoise_output_type(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23Denoise
        assert RCLTX23Denoise.RETURN_TYPES == ("IMAGE",)

    def test_denoise_optional_params(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23Denoise
        inputs = RCLTX23Denoise.INPUT_TYPES()
        opt = inputs.get("optional", {})
        assert "steps" in opt
        assert "noise_scale" in opt

    def test_rcltx23_no_resolution_dropdown(self):
        """RCLTX23 should no longer expose a resolution input."""
        from comfyui_nodes.rc_models import RCLTX23
        inputs = RCLTX23.INPUT_TYPES()
        all_keys = set(inputs.get("required", {}).keys()) | set(inputs.get("optional", {}).keys())
        assert "resolution" not in all_keys


# ── Pipeline method signature tests ─────────────────────────────────


class TestPipelineMethodSignatures:
    """LTX23Pipeline must expose the 3 composable public methods."""

    def test_generate_latent_method_exists(self):
        from src.pipelines.ltx2 import LTX23Pipeline
        assert hasattr(LTX23Pipeline, "generate_latent")
        assert callable(getattr(LTX23Pipeline, "generate_latent"))

    def test_latent_upscale_method_exists(self):
        from src.pipelines.ltx2 import LTX23Pipeline
        assert hasattr(LTX23Pipeline, "latent_upscale")
        assert callable(getattr(LTX23Pipeline, "latent_upscale"))

    def test_denoise_latent_method_exists(self):
        from src.pipelines.ltx2 import LTX23Pipeline
        assert hasattr(LTX23Pipeline, "denoise_latent")
        assert callable(getattr(LTX23Pipeline, "denoise_latent"))

    def test_generate_latent_signature(self):
        import inspect
        from src.pipelines.ltx2 import LTX23Pipeline
        sig = inspect.signature(LTX23Pipeline.generate_latent)
        params = list(sig.parameters.keys())
        for expected in ["prompt", "negative_prompt", "width", "height",
                         "num_inference_steps", "guidance_scale", "generator",
                         "control_image", "num_frames", "fps"]:
            assert expected in params, f"Missing param: {expected}"

    def test_denoise_latent_signature(self):
        import inspect
        from src.pipelines.ltx2 import LTX23Pipeline
        sig = inspect.signature(LTX23Pipeline.denoise_latent)
        params = list(sig.parameters.keys())
        for expected in ["video_latents", "audio_latents", "prompt",
                         "width", "height", "num_inference_steps", "sigma_values"]:
            assert expected in params, f"Missing param: {expected}"


# ── Example workflow JSON validity ──────────────────────────────────


class TestWorkflowValidity:
    """All example workflow JSONs must reference only registered nodes."""

    WORKFLOW_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "example_workflows",
    )

    @pytest.fixture
    def node_registry(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS
        return set(NODE_CLASS_MAPPINGS.keys())

    @pytest.fixture
    def comfyui_builtins(self):
        """Common ComfyUI built-in nodes that won't be in our registry."""
        return {"SaveImage", "LoadImage", "PreviewImage", "LatentUpscale"}

    def _check_workflow(self, filename, node_registry, comfyui_builtins):
        path = os.path.join(self.WORKFLOW_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"{filename} not found")
        with open(path) as f:
            wf = json.load(f)
        for node_id, node in wf.items():
            ct = node.get("class_type", "")
            assert ct in node_registry or ct in comfyui_builtins, (
                f"{filename} node {node_id} references unknown class_type '{ct}'"
            )

    def test_ltx23_minimal(self, node_registry, comfyui_builtins):
        self._check_workflow("rc_ltx23_minimal.json", node_registry, comfyui_builtins)

    def test_ltx23_high_minimal(self, node_registry, comfyui_builtins):
        self._check_workflow("rc_ltx23_high_minimal.json", node_registry, comfyui_builtins)


# ── Node execution with mocked pipeline ─────────────────────────────


class TestNodeExecution:
    """Test actual node execution with a mocked LTX23Pipeline."""

    def _make_mock_pipe(self):
        """Create a mock pipeline with the 3 public methods."""
        pipe = Mock()

        # generate_latent returns a dict matching LTX23_LATENT
        pipe.generate_latent.return_value = {
            "video_latents": torch.randn(1, 128, 6, 8, 12),
            "audio_latents": torch.randn(1, 64, 10),
            "audio_tensor": torch.randn(1, 48000),
            "audio_sample_rate": 24000,
            "prompt": "test prompt",
            "negative_prompt": "",
            "width": 768,
            "height": 512,
            "num_frames": 41,
            "fps": 24,
        }

        # latent_upscale returns upscaled tensor (2x spatial)
        pipe.latent_upscale.return_value = torch.randn(1, 128, 6, 16, 24)

        # denoise_latent returns video_tensor
        pipe.denoise_latent.return_value = {
            "video_tensor": torch.randint(0, 255, (41, 1024, 1536, 3), dtype=torch.uint8),
        }

        return pipe

    def test_generate_latent_node(self, monkeypatch):
        from comfyui_nodes.rc_latent_workflow import RCLTX23GenerateLatent

        # Mock comfy_pipeline_observer as no-op context manager
        mock_observer = MagicMock()
        mock_observer.return_value.__enter__ = Mock(return_value=None)
        mock_observer.return_value.__exit__ = Mock(return_value=False)
        monkeypatch.setattr(
            "comfyui_nodes.rc_latent_workflow.RCLTX23GenerateLatent.generate.__module__",
            "comfyui_nodes.rc_latent_workflow",
            raising=False,
        )

        pipe = self._make_mock_pipe()
        node = RCLTX23GenerateLatent()

        with patch("src.pipelines.comfy_callbacks.comfy_pipeline_observer", mock_observer):
            (result,) = node.generate(
                pipe=pipe,
                prompt="test prompt",
                width=768,
                height=512,
                steps=25,
                cfg=4.0,
                seed=42,
            )

        assert "video_latents" in result
        assert "audio_latents" in result
        assert result["width"] == 768
        assert result["height"] == 512
        pipe.generate_latent.assert_called_once()

    def test_latent_upscale_node(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23LatentUpscale

        pipe = self._make_mock_pipe()
        input_latent = pipe.generate_latent.return_value

        node = RCLTX23LatentUpscale()
        (result,) = node.upscale(pipe=pipe, ltx_latent=input_latent)

        assert result["width"] == 768 * 2
        assert result["height"] == 512 * 2
        assert result["video_latents"].shape != input_latent["video_latents"].shape
        pipe.latent_upscale.assert_called_once()

    def test_denoise_node(self, monkeypatch):
        from comfyui_nodes.rc_latent_workflow import RCLTX23Denoise

        mock_observer = MagicMock()
        mock_observer.return_value.__enter__ = Mock(return_value=None)
        mock_observer.return_value.__exit__ = Mock(return_value=False)

        pipe = self._make_mock_pipe()
        upscaled_latent = {
            "video_latents": torch.randn(1, 128, 6, 16, 24),
            "audio_latents": torch.randn(1, 64, 10),
            "prompt": "test prompt",
            "negative_prompt": "",
            "width": 1536,
            "height": 1024,
            "num_frames": 41,
            "fps": 24,
        }

        node = RCLTX23Denoise()

        with patch("src.pipelines.comfy_callbacks.comfy_pipeline_observer", mock_observer):
            (frames,) = node.denoise(pipe=pipe, ltx_latent=upscaled_latent)

        # Output should be ComfyUI IMAGE: float32 [B,H,W,C] in [0,1]
        assert frames.dtype == torch.float32
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0
        assert frames.ndim == 4  # [T, H, W, C]
        pipe.denoise_latent.assert_called_once()

    def test_full_3_node_chain(self, monkeypatch):
        """Simulate the full GenerateLatent → LatentUpscale → Denoise chain."""
        from comfyui_nodes.rc_latent_workflow import (
            RCLTX23GenerateLatent,
            RCLTX23LatentUpscale,
            RCLTX23Denoise,
        )

        mock_observer = MagicMock()
        mock_observer.return_value.__enter__ = Mock(return_value=None)
        mock_observer.return_value.__exit__ = Mock(return_value=False)

        pipe = self._make_mock_pipe()

        with patch("src.pipelines.comfy_callbacks.comfy_pipeline_observer", mock_observer):
            # Stage 1
            (latent,) = RCLTX23GenerateLatent().generate(
                pipe=pipe, prompt="test", width=768, height=512,
                steps=25, cfg=4.0, seed=42,
            )

        assert latent["width"] == 768
        assert latent["height"] == 512

        # Stage 2
        (upscaled,) = RCLTX23LatentUpscale().upscale(pipe=pipe, ltx_latent=latent)
        assert upscaled["width"] == 1536
        assert upscaled["height"] == 1024

        # Stage 3
        with patch("src.pipelines.comfy_callbacks.comfy_pipeline_observer", mock_observer):
            (frames,) = RCLTX23Denoise().denoise(pipe=pipe, ltx_latent=upscaled)

        assert frames.dtype == torch.float32
        assert frames.ndim == 4

    def test_generate_latent_rejects_wrong_pipeline(self):
        """Should raise if pipe lacks generate_latent method."""
        from comfyui_nodes.rc_latent_workflow import RCLTX23GenerateLatent

        bad_pipe = Mock(spec=[])  # no methods
        node = RCLTX23GenerateLatent()

        with pytest.raises(ValueError, match="LTX-2.3 pipeline"):
            node.generate(pipe=bad_pipe, prompt="x", width=512, height=512,
                          steps=1, cfg=1.0, seed=0)

    def test_latent_upscale_rejects_wrong_pipeline(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23LatentUpscale

        bad_pipe = Mock(spec=[])
        node = RCLTX23LatentUpscale()

        with pytest.raises(ValueError, match="LTX-2.3 pipeline"):
            node.upscale(pipe=bad_pipe, ltx_latent={})

    def test_denoise_rejects_wrong_pipeline(self):
        from comfyui_nodes.rc_latent_workflow import RCLTX23Denoise

        bad_pipe = Mock(spec=[])
        node = RCLTX23Denoise()

        with pytest.raises(ValueError, match="LTX-2.3 pipeline"):
            node.denoise(pipe=bad_pipe, ltx_latent={})
