"""CPU-only registration and config guards for the Anima pipeline.

These must pass on a machine with NO ai-toolkit checkout, no model weights, and
a diffusers that predates the Anima module: src/api/v1/inference.py imports
PIPELINE_REGISTRY at module scope and _build_pipeline_registry() has no
try/except, so a module-level `from diffusers import AnimaAutoBlocks` in
src/pipelines/anima.py would break FastAPI startup for the whole catalog on any
container whose diffusers is older than c943837.

Anima's actual generation (the modular pipeline, the LoRA fuse, sample parity)
can only be exercised on a GPU with weights and diffusers >= c943837 -- that is
the release gate, not these tests.
"""

import importlib

import pytest

from src.pipelines import get_pipeline_class, get_pipeline_config
from src.schemas.models import ModelType, get_supported_models
from src.services.download_config import get_download_config


class TestAnimaRegistration:
    def test_model_type_exists(self):
        assert "anima" in get_supported_models()
        assert ModelType.ANIMA.value == "anima"

    def test_pipeline_class_resolves(self):
        cls = get_pipeline_class("anima")
        assert cls is not None
        assert cls.__name__ == "AnimaPipeline"
        assert cls.CONFIG.model_type is ModelType.ANIMA

    def test_registry_builds_and_contains_anima(self):
        """PIPELINE_REGISTRY is built eagerly by src/api/v1/inference.py."""
        from src.pipelines import PIPELINE_REGISTRY

        assert ModelType.ANIMA in PIPELINE_REGISTRY
        assert PIPELINE_REGISTRY[ModelType.ANIMA] is get_pipeline_class("anima")

    def test_id_mirrors_training_arch_name(self):
        """ai-toolkit arch 'anima' -> model id 'anima' (anima.json schema)."""
        assert "anima" in get_supported_models()


class TestAnimaImportSafety:
    """The heavy / version-sensitive imports must be method-scoped: importing the
    module and reading CONFIG must not require diffusers to have the Anima classes
    or ai-toolkit to be installed."""

    def test_module_imports_without_diffusers_anima(self):
        # Importing the module executes only its top-level imports (torch, stdlib,
        # .base, ..schemas). If any diffusers-Anima / toolkit symbol were imported
        # at module scope, this would raise on an older diffusers.
        mod = importlib.import_module("src.pipelines.anima")
        assert hasattr(mod, "AnimaPipeline")

    def test_config_readable_without_instantiation(self):
        cfg = get_pipeline_config("anima")
        assert cfg is not None
        assert cfg.base_model == "circlestone-labs/Anima-Base-v1.0-Diffusers"


class TestAnimaConfig:
    def test_config_values(self):
        cfg = get_pipeline_config("anima")
        assert cfg.base_model == "circlestone-labs/Anima-Base-v1.0-Diffusers"
        assert cfg.default_steps == 30  # defaultSamples.ts:48
        assert cfg.default_guidance_scale == 4.0  # defaultSamples.ts:47
        assert cfg.requires_control_image is False
        assert cfg.supports_negative_prompt is True
        assert cfg.is_video_model is False

    def test_resolution_divisor_is_32(self):
        """The trainer's get_bucket_divisibility() = 16*2 = 32 and
        generate_single_image rounds preview W/H to it (anima.py:246, 438-440).
        The modular pipeline only requires 16, so 32 satisfies both."""
        assert get_pipeline_config("anima").resolution_divisor == 32

    def test_lora_merge_method_is_custom(self):
        """Anima fuses the LoRA itself during _load_pipeline (via the
        AnimaLoraLoaderMixin), so switching scale needs a reload."""
        from src.pipelines.base import LoraMergeMethod

        assert get_pipeline_config("anima").lora_merge_method is LoraMergeMethod.CUSTOM

    def test_default_negative_is_the_anima_preset(self):
        """anima.py preset (options.ts:84-87) fills a quality-negative prompt by
        default; an empty request neg should fall back to it, not to ''."""
        neg = get_pipeline_config("anima").default_neg
        assert "worst quality" in neg and "jpeg artifacts" in neg


class TestAnimaGuidanceIsRaw:
    """Anima uses RAW guidance -- the trainer sets guider.guidance_scale =
    gen_config.guidance_scale with NO -1 (anima.py:444). This is the OPPOSITE of
    Krea 2, which 0-normalizes with max(0, g-1). If anima.py ever grows a
    `max(0, g-1)` it under-guides by a full point; this pins that it must not."""

    def test_source_has_no_zero_normalization(self):
        import inspect

        import src.pipelines.anima as A

        src = inspect.getsource(A.AnimaPipeline._run_inference)
        assert "guidance_scale - 1" not in src
        assert "guidance_scale-1" not in src
        # It must forward the raw value straight to the guider.
        assert "self.pipe.guider.guidance_scale = float(guidance_scale)" in src


class TestAnimaDownloads:
    def test_full_repo_pull_no_filters(self):
        """circlestone-labs/Anima-Base-v1.0-Diffusers is self-contained: every
        component maps back to a subfolder of this one repo, so we pull it whole
        with no extras."""
        cfg = get_download_config(ModelType.ANIMA)
        assert cfg.allow_patterns is None
        assert cfg.ignore_patterns is None
        assert cfg.extras == []


class TestAnimaComfyNodes:
    def test_node_registered(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        assert "RCAnima" in NODE_CLASS_MAPPINGS
        assert "RCAnima" in NODE_DISPLAY_NAME_MAPPINGS

    def test_node_defaults_match_pipeline_config(self):
        """A ComfyUI widget offering a different step/guidance/resolution grid
        than the pipeline default is a silent parity bug."""
        from comfyui_nodes import NODE_CLASS_MAPPINGS

        cls = NODE_CLASS_MAPPINGS["RCAnima"]
        cfg = get_pipeline_config(cls.MODEL_ID)
        assert cls.MODEL_ID == "anima"
        assert cls.RESOLUTION_STEP == cfg.resolution_divisor == 32
        assert cls.DEFAULT_STEPS == cfg.default_steps == 30
        assert cls.DEFAULT_GUIDANCE == cfg.default_guidance_scale == 4.0
        assert cls.REQUIRES_CONTROL_IMAGE is cfg.requires_control_image is False

    def test_latent_workflow_lists_and_ctor_map_stay_in_sync(self):
        """ALL_PIPELINES and ctor_map are two hand-maintained lists; a model in
        the first but not the second raises ValueError at execution time."""
        import inspect

        from comfyui_nodes.rc_latent_workflow import RCAITKLoadPipeline

        src = inspect.getsource(RCAITKLoadPipeline)
        assert "anima" in RCAITKLoadPipeline.ALL_PIPELINES
        assert '"anima": lambda' in src
