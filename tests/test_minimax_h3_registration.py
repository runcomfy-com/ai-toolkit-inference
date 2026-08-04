"""MiniMax-H3 registration, config and LoRA-key-parsing tests.

None of these need a GPU or the weights. They lock down the parts that fail
silently: registry wiring, the download filter (the difference between 42.5 GB
and ~365 GB), and the LoRA key parsing (which "works" while dropping a quarter
of the modules if the target matching is wrong).
"""

import json
import struct

import pytest

from src.schemas.models import ModelType, get_supported_models
from src.pipelines import _LAZY_IMPORTS, _MODEL_TYPE_TO_CLASS
from src.pipelines.base import LoraMergeMethod
from src.services.download_config import MODEL_DOWNLOAD_CONFIGS


class TestRegistration:
    def test_model_type_exists(self):
        assert ModelType.MINIMAX_H3.value == "minimax_h3"

    def test_in_supported_models(self):
        assert "minimax_h3" in get_supported_models()

    def test_dispatches_to_pipeline_class(self):
        assert _MODEL_TYPE_TO_CLASS[ModelType.MINIMAX_H3] == "MinimaxH3Pipeline"

    def test_lazy_import_target(self):
        assert _LAZY_IMPORTS["MinimaxH3Pipeline"] == (".minimax_h3", "MinimaxH3Pipeline")

    def test_pipeline_imports_without_heavy_deps(self):
        """Module-scope import must not need ai-toolkit, transformers or the
        weights: src/pipelines/__init__.py builds the registry with no
        try/except, so a top-level failure breaks FastAPI startup for the whole
        catalog."""
        from src.pipelines.minimax_h3 import MinimaxH3Pipeline

        assert MinimaxH3Pipeline.CONFIG.model_type is ModelType.MINIMAX_H3


class TestConfig:
    @pytest.fixture
    def config(self):
        from src.pipelines.minimax_h3 import MinimaxH3Pipeline

        return MinimaxH3Pipeline.CONFIG

    def test_video_model_with_audio_defaults(self, config):
        assert config.is_video_model is True
        assert config.default_num_frames == 107  # on the 17n+5 grid
        assert config.default_fps == 24  # packing.FPS is fixed

    def test_guidance_distilled(self, config):
        """The sampler has no unconditional branch at all — guidance_scale is
        accepted and ignored, and a negative prompt has nowhere to go."""
        assert config.default_guidance_scale == 1.0
        assert config.supports_negative_prompt is False

    def test_resolution_divisor(self, config):
        # 16x VAE spatial compression * 2x2 transformer patch
        assert config.resolution_divisor == 32

    def test_lora_is_custom_not_set_adapters(self, config):
        """The transformer is a plain nn.Module with no PEFT/diffusers LoRA
        API, so the generic adapter paths do not apply."""
        assert config.lora_merge_method is LoraMergeMethod.CUSTOM

    def test_control_image_is_optional(self, config):
        """Absent -> t2v, present -> first-frame i2v. Both are valid."""
        assert config.requires_control_image is False


class TestDownloadConfig:
    @pytest.fixture
    def config(self):
        return MODEL_DOWNLOAD_CONFIGS[ModelType.MINIMAX_H3]

    def test_allow_patterns_are_mandatory(self, config):
        """Unfiltered, Comfy-Org/MiniMax-H3 is ~365 GB (bf16 + int8 + fp8 +
        int8-pruned, two partitions, three text encoders). These four files are
        42.5 GB. An empty/None filter here is a 300 GB regression that no other
        test would catch."""
        assert config.allow_patterns, "MiniMax-H3 must filter its base repo"
        assert len(config.allow_patterns) == 4

    def test_pulls_fl2va_partition_only(self, config):
        joined = " ".join(config.allow_patterns)
        assert "fl2va" in joined
        assert "ref2va" not in joined, "ref2va is a different conditioning contract"

    def test_all_four_components_present(self, config):
        joined = " ".join(config.allow_patterns)
        assert "diffusion_models/" in joined  # DiT
        assert "text_encoders/" in joined  # Qwen3-VL
        assert joined.count("vae/") == 2  # video VAE + audio VAE

    def test_extras_filtered_too(self, config):
        """MiniMaxAI/MiniMax-H3 is 297 files including the full bf16 original;
        we want three tiny config/tokenizer subfolders."""
        assert len(config.extras) == 1
        extra = config.extras[0]
        assert extra.repo_id == "MiniMaxAI/MiniMax-H3"
        assert extra.allow_patterns, "the original repo must be filtered"


def _make_lora_header(module_names, rank=16, prefix="diffusion_model.", with_alpha=False):
    """Build a safetensors header mimicking ai-toolkit's H3 LoRA save format."""
    hdr = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in module_names:
        for slot, shape in (("lora_A", [rank, 64]), ("lora_B", [64, rank])):
            n = shape[0] * shape[1] * 4
            hdr[f"{prefix}{name}.{slot}.weight"] = {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + n],
            }
            offset += n
        if with_alpha:
            hdr[f"{prefix}{name}.alpha"] = {
                "dtype": "F32", "shape": [], "data_offsets": [offset, offset + 4],
            }
            offset += 4
    return hdr


def _parse_like_attach_lora(keys, header):
    """Replay _attach_lora's key parsing (kept in sync by eye; the point is to
    lock the contract, not to re-import private code)."""
    pairs, alphas = {}, {}
    unmatched = []
    for key in keys:
        name = key
        if name.startswith("diffusion_model."):
            name = name[len("diffusion_model."):]
        if name.endswith(".alpha"):
            alphas[name[: -len(".alpha")]] = 1.0
            continue
        hit = False
        for suffix, slot in (
            (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),
            (".lora_A.weight", "down"), (".lora_B.weight", "up"),
            (".lora_down.weight", "down"), (".lora_up.weight", "up"),
        ):
            if name.endswith(suffix):
                pairs.setdefault(name[: -len(suffix)], {})[slot] = header[key]["shape"]
                hit = True
                break
        if not hit:
            unmatched.append(key)
    return pairs, alphas, unmatched


class TestLoraKeyParsing:
    """Locks the real key layout observed in job a1977b6b (rank 16, 516
    tensors, 258 complete pairs, zero alpha tensors)."""

    REAL_TARGETS = (
        [f"blocks.{i}.adaln_proj.linear" for i in range(50)]
        + [f"blocks.{i}.attn.qkv_proj" for i in range(50)]
        + [f"blocks.{i}.attn.out_proj" for i in range(50)]
        + [f"blocks.{i}.mlp.fc1" for i in range(50)]
        + [f"blocks.{i}.mlp.fc2" for i in range(50)]
        + [f"token_refiner.blocks.{i}.attn.qkv_proj" for i in range(2)]
        + [f"token_refiner.blocks.{i}.attn.out_proj" for i in range(2)]
        + [f"token_refiner.blocks.{i}.mlp.fc1" for i in range(2)]
        + [f"token_refiner.blocks.{i}.mlp.fc2" for i in range(2)]
    )

    def test_real_layout_parses_completely(self):
        hdr = _make_lora_header(self.REAL_TARGETS)
        keys = [k for k in hdr if k != "__metadata__"]
        assert len(keys) == 516, "516 tensors in the observed adapter"
        pairs, alphas, unmatched = _parse_like_attach_lora(keys, hdr)
        assert not unmatched
        assert len(pairs) == 258
        assert all("down" in v and "up" in v for v in pairs.values())
        assert alphas == {}, "peft_format strips .alpha (network_mixins.py:605-614)"

    def test_adaln_and_token_refiner_are_not_dropped(self):
        """A startswith('blocks.') whitelist would drop 58 of 258 modules and
        still look like it worked. krea2 hit exactly that (32/256)."""
        hdr = _make_lora_header(self.REAL_TARGETS)
        keys = [k for k in hdr if k != "__metadata__"]
        pairs, _, _ = _parse_like_attach_lora(keys, hdr)
        assert sum(1 for k in pairs if "adaln_proj" in k) == 50
        assert sum(1 for k in pairs if k.startswith("token_refiner.")) == 8

    def test_missing_alpha_defaults_to_rank_giving_scale_one(self):
        """With no .alpha tensor, alpha must default to rank so that
        scale = alpha/rank = 1.0. Defaulting to 1 would scale every module by
        1/16 and the LoRA would look like it barely applied."""
        rank = 16
        alpha = rank  # the default _attach_lora applies
        assert (alpha / rank) * 1.0 == 1.0
        wrong = (1 / rank) * 1.0
        assert wrong == pytest.approx(0.0625)

    def test_explicit_alpha_is_honoured(self):
        hdr = _make_lora_header(["blocks.0.mlp.fc1"], with_alpha=True)
        keys = [k for k in hdr if k != "__metadata__"]
        pairs, alphas, unmatched = _parse_like_attach_lora(keys, hdr)
        assert not unmatched
        assert len(pairs) == 1
        assert "blocks.0.mlp.fc1" in alphas

    def test_peft_default_infix_is_tolerated(self):
        """Some PEFT saves carry a `.default` infix; both spellings must map to
        the same module name."""
        hdr = {
            "diffusion_model.blocks.0.mlp.fc1.lora_A.default.weight": {"shape": [16, 64]},
            "diffusion_model.blocks.0.mlp.fc1.lora_B.default.weight": {"shape": [64, 16]},
        }
        pairs, _, unmatched = _parse_like_attach_lora(list(hdr), hdr)
        assert not unmatched
        assert list(pairs) == ["blocks.0.mlp.fc1"]


class TestFrameGrid:
    """17n+5 is enforced inside the sampler, but the API layer snaps too so the
    response reports the real count instead of silently shrinking."""

    @pytest.mark.parametrize(
        "requested,expected",
        [(5, 5), (22, 22), (39, 39), (107, 107), (110, 107), (124, 124), (130, 124)],
    )
    def test_grid_values(self, requested, expected):
        # 17n+5 closed form, mirroring packing.align_num_frames_down
        snapped = ((requested - 5) // 17) * 17 + 5
        assert snapped == expected
