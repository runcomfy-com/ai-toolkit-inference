"""CPU-only registration and config guards for the Krea 2 pipelines.

These must pass on a machine with NO ai-toolkit checkout and no model weights:
src/api/v1/inference.py imports PIPELINE_REGISTRY at module scope and
_build_pipeline_registry() has no try/except, so a module-level import in
src/pipelines/krea2.py would break FastAPI startup for the whole catalog.
"""

import pytest

from src.pipelines import get_pipeline_class, get_pipeline_config
from src.schemas.models import ModelType, get_supported_models
from src.services.download_config import get_download_config


KREA2_MODELS = ["krea2", "krea2_turbo", "krea2_o_edit", "krea2_o_edit_turbo"]

# (model_id, base_model, checkpoint, steps, guidance, requires_ctrl, is_edit)
EXPECTED = [
    ("krea2", "krea/Krea-2-Raw", "raw.safetensors", 30, 4.0, False, False),
    ("krea2_turbo", "krea/Krea-2-Turbo", "turbo.safetensors", 9, 1.0, False, False),
    ("krea2_o_edit", "krea/Krea-2-Raw", "raw.safetensors", 30, 4.0, True, True),
    ("krea2_o_edit_turbo", "krea/Krea-2-Turbo", "turbo.safetensors", 8, 1.0, True, True),
]


class TestKrea2Registration:
    """The model must be reachable through every registry the server uses."""

    @pytest.mark.parametrize("model_id", KREA2_MODELS)
    def test_model_type_exists(self, model_id):
        assert model_id in get_supported_models()
        assert ModelType(model_id).value == model_id

    @pytest.mark.parametrize("model_id", KREA2_MODELS)
    def test_pipeline_class_resolves(self, model_id):
        """Importing the class must not need ai-toolkit or diffusers present."""
        cls = get_pipeline_class(model_id)
        assert cls is not None, f"{model_id} has no pipeline class"
        assert cls.CONFIG.model_type.value == model_id

    def test_registry_builds_without_ai_toolkit(self):
        """PIPELINE_REGISTRY is built eagerly by src/api/v1/inference.py."""
        from src.pipelines import PIPELINE_REGISTRY

        for model_id in KREA2_MODELS:
            assert ModelType(model_id) in PIPELINE_REGISTRY

    def test_ids_mirror_training_arch_names(self):
        """ai-toolkit arch 'krea2:o_edit' -> model id 'krea2_o_edit'."""
        for arch in ["krea2", "krea2:turbo", "krea2:o_edit", "krea2:o_edit_turbo"]:
            assert arch.replace(":", "_") in get_supported_models()


class TestKrea2Config:
    @pytest.mark.parametrize(
        "model_id,base_model,checkpoint,steps,guidance,requires_ctrl,is_edit", EXPECTED
    )
    def test_config_values(
        self, model_id, base_model, checkpoint, steps, guidance, requires_ctrl, is_edit
    ):
        cls = get_pipeline_class(model_id)
        cfg = cls.CONFIG
        assert cfg.base_model == base_model
        assert cls.CHECKPOINT_FILENAME == checkpoint
        assert cfg.default_steps == steps
        assert cfg.default_guidance_scale == guidance
        assert cfg.requires_control_image is requires_ctrl
        assert cls.IS_EDIT is is_edit

    @pytest.mark.parametrize("model_id", KREA2_MODELS)
    def test_resolution_divisor_is_16(self, model_id):
        """VAE f8 x patch 2. Anything else silently re-snaps in the sampler."""
        assert get_pipeline_config(model_id).resolution_divisor == 16

    @pytest.mark.parametrize("model_id", KREA2_MODELS)
    def test_lora_merge_method_is_custom(self, model_id):
        """Krea 2 has no load_lora_weights/fuse_lora; merging is manual."""
        from src.pipelines.base import LoraMergeMethod

        assert get_pipeline_config(model_id).lora_merge_method is LoraMergeMethod.CUSTOM

    @pytest.mark.parametrize("model_id", KREA2_MODELS)
    def test_not_a_video_model(self, model_id):
        assert get_pipeline_config(model_id).is_video_model is False

    def test_edit_variants_enable_kv_cache_and_match_target_res(self):
        """Both edit presets train with model_kwargs {edit, match_target_res,
        kv_cache} all true. kv_cache changes the attention mask, so a LoRA
        trained with it on must be inferenced with it on."""
        for model_id in ("krea2_o_edit", "krea2_o_edit_turbo"):
            cls = get_pipeline_class(model_id)
            assert cls.USE_KV_CACHE is True
            assert cls.MATCH_TARGET_RES is True

    def test_t2i_variants_disable_edit_paths(self):
        for model_id in ("krea2", "krea2_turbo"):
            cls = get_pipeline_class(model_id)
            assert cls.USE_KV_CACHE is False
            assert cls.MATCH_TARGET_RES is False


class TestKrea2Downloads:
    @pytest.mark.parametrize(
        "model_id,checkpoint",
        [(m, c) for m, _, c, *_ in EXPECTED],
    )
    def test_allow_patterns_pin_the_single_file_checkpoint(self, model_id, checkpoint):
        """The krea repos also ship a full diffusers layout we never load."""
        cfg = get_download_config(ModelType(model_id))
        assert cfg.allow_patterns == [checkpoint]

    @pytest.mark.parametrize("model_id", KREA2_MODELS)
    def test_extras_cover_text_encoder_and_vae(self, model_id):
        cfg = get_download_config(ModelType(model_id))
        repos = {e.repo_id for e in cfg.extras}
        assert "Qwen/Qwen3-VL-4B-Instruct" in repos
        assert "Qwen/Qwen-Image" in repos

    def test_shared_repos_declare_identical_allow_patterns(self):
        """cli/download_models.py:_merge_download_tasks dedupes by repo_id and
        collapses to an unfiltered full-repo pull if the patterns disagree."""
        by_repo = {}
        for model_id in KREA2_MODELS:
            for extra in get_download_config(ModelType(model_id)).extras:
                by_repo.setdefault(extra.repo_id, set()).add(
                    tuple(extra.allow_patterns) if extra.allow_patterns else None
                )
        for repo_id, pattern_sets in by_repo.items():
            assert len(pattern_sets) == 1, (
                f"{repo_id} declared with differing allow_patterns across krea2 "
                f"models: {pattern_sets} -- this collapses to a full-repo download"
            )


class TestKrea2ComfyNodes:
    def test_latent_workflow_lists_and_ctor_map_stay_in_sync(self):
        """ALL_PIPELINES and ctor_map are two hand-maintained lists; a model in
        the first but not the second raises ValueError at execution time."""
        import inspect

        from comfyui_nodes.rc_latent_workflow import RCAITKLoadPipeline

        src = inspect.getsource(RCAITKLoadPipeline)
        for model_id in KREA2_MODELS:
            assert f'"{model_id}"' in RCAITKLoadPipeline.ALL_PIPELINES or (
                model_id in RCAITKLoadPipeline.ALL_PIPELINES
            ), f"{model_id} missing from ALL_PIPELINES"
            # ctor_map lives inside the load() method body
            assert f'"{model_id}": lambda' in src, f"{model_id} missing from ctor_map"

    def test_nodes_registered(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        for node in ("RCKrea2", "RCKrea2Turbo", "RCKrea2Edit", "RCKrea2EditTurbo"):
            assert node in NODE_CLASS_MAPPINGS
            assert node in NODE_DISPLAY_NAME_MAPPINGS

    def test_node_defaults_match_pipeline_config(self):
        """A ComfyUI widget offering a different step/guidance/resolution grid
        than the pipeline default is a silent parity bug."""
        from comfyui_nodes import NODE_CLASS_MAPPINGS

        for node in ("RCKrea2", "RCKrea2Turbo", "RCKrea2Edit", "RCKrea2EditTurbo"):
            cls = NODE_CLASS_MAPPINGS[node]
            cfg = get_pipeline_config(cls.MODEL_ID)
            assert cls.RESOLUTION_STEP == cfg.resolution_divisor
            assert cls.DEFAULT_STEPS == cfg.default_steps
            assert cls.DEFAULT_GUIDANCE == cfg.default_guidance_scale
            assert cls.REQUIRES_CONTROL_IMAGE is cfg.requires_control_image


class TestKrea2GuidanceNormalization:
    """The trainer does `guidance = max(0, sample.guidance_scale - 1)` before
    sampling, so the request-level guidance_scale means the same thing in both
    places. Getting this wrong is a silent 33% over-guidance at cfg=4."""

    @pytest.mark.parametrize(
        "user_cfg,internal", [(0.0, 0.0), (1.0, 0.0), (4.0, 3.0), (7.5, 6.5)]
    )
    def test_normalization_formula(self, user_cfg, internal):
        assert max(0.0, float(user_cfg) - 1.0) == pytest.approx(internal)

    def test_turbo_default_disables_cfg(self):
        """cfg 1.0 -> internal 0.0 -> uncond pass skipped entirely."""
        for model_id in ("krea2_turbo", "krea2_o_edit_turbo"):
            cfg = get_pipeline_config(model_id)
            assert max(0.0, cfg.default_guidance_scale - 1.0) == 0.0
            assert cfg.supports_negative_prompt is False

    def test_raw_default_enables_cfg(self):
        for model_id in ("krea2", "krea2_o_edit"):
            cfg = get_pipeline_config(model_id)
            assert max(0.0, cfg.default_guidance_scale - 1.0) > 0.0
            assert cfg.supports_negative_prompt is True


class TestKrea2ComfyModelOptions:
    """The HTTP API and the ComfyUI nodes are two independent entry points.

    Adding kv_cache / match_target_res to InferenceInput fixed only the HTTP
    path; the ComfyUI nodes kept using the hardcoded class defaults, so an edit
    adapter trained before ai-toolkit's 2026-07-16 preset change silently
    produced wrong output through ComfyUI. These tests pin both halves.
    """

    def test_edit_nodes_expose_the_options(self):
        from comfyui_nodes import NODE_CLASS_MAPPINGS

        for node in ("RCKrea2Edit", "RCKrea2EditTurbo"):
            spec = NODE_CLASS_MAPPINGS[node].INPUT_TYPES()
            opt = spec["optional"]
            for name in ("kv_cache", "match_target_res"):
                assert name in opt, f"{node} does not expose {name}"
                typ, cfg = opt[name]
                assert typ == "BOOLEAN"
                assert cfg["default"] is True, "default must follow the current preset"
                assert cfg.get("tooltip"), f"{node}.{name} needs a tooltip"

    def test_t2i_nodes_do_not_expose_them(self):
        """They are meaningless without reference images -- do not clutter the UI."""
        from comfyui_nodes import NODE_CLASS_MAPPINGS

        for node in ("RCKrea2", "RCKrea2Turbo"):
            opt = NODE_CLASS_MAPPINGS[node].INPUT_TYPES()["optional"]
            assert "kv_cache" not in opt
            assert "match_target_res" not in opt

    def test_generate_forwards_options_to_the_pipeline(self):
        """The node must actually call apply_model_options -- declaring the
        inputs without wiring them is the exact bug this guards."""
        from comfyui_nodes import NODE_CLASS_MAPPINGS

        seen = {}

        class FakePipe:
            CONFIG = None

            def apply_model_options(self, **opts):
                seen.update(opts)
                seen["_called"] = True

            def generate(self, **kw):
                from PIL import Image

                return {"image": Image.new("RGB", (16, 16))}

        node = NODE_CLASS_MAPPINGS["RCKrea2Edit"]()
        import comfyui_nodes.rc_models as rc

        orig = rc.get_or_load_pipeline
        rc.get_or_load_pipeline = lambda **kw: FakePipe()
        try:
            node.generate(
                prompt="x", width=1024, height=1024, sample_steps=8,
                guidance_scale=4.0, seed=1, kv_cache=False, match_target_res=False,
            )
        finally:
            rc.get_or_load_pipeline = orig

        assert seen.get("_called"), "apply_model_options was never called"
        assert seen["kv_cache"] is False
        assert seen["match_target_res"] is False

    def test_latent_workflow_generate_exposes_and_forwards(self):
        from comfyui_nodes.rc_latent_workflow import RCAITKGenerate

        opt = RCAITKGenerate.INPUT_TYPES()["optional"]
        assert "kv_cache" in opt and "match_target_res" in opt

        seen = {}

        class FakePipe:
            CONFIG = None

            def apply_model_options(self, **o):
                seen.update(o)

            def generate(self, **kw):
                from PIL import Image

                return {"image": Image.new("RGB", (16, 16))}

        RCAITKGenerate().generate(
            pipe=FakePipe(), prompt="x", width=1024, height=1024, steps=8,
            cfg=4.0, seed=1, kv_cache=False, match_target_res=False,
        )
        assert seen == {"kv_cache": False, "match_target_res": False}

    def test_apply_model_options_is_safe_for_models_without_any(self):
        """RCAITKGenerate calls it unconditionally, so every other pipeline
        must tolerate the call."""
        from src.pipelines import get_pipeline_class

        for model_id in ("flux", "sdxl", "qwen_image", "ltx2", "wan22_5b"):
            cls = get_pipeline_class(model_id)
            inst = object.__new__(cls)
            assert inst.apply_model_options(kv_cache=False, match_target_res=True) is None
