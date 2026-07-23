"""CPU functional tests for the Krea 2 LoRA merge and the flow-matching sampler.

These build a tiny SingleStreamDiT (~0.7M params instead of 12.8B) so the real
upstream modules are exercised without downloading any weights. They skip
automatically when no ai-toolkit checkout is reachable.

The LoRA test exists specifically to catch a silent-truncation bug: ai-toolkit's
module gate is a substring test for "blocks", and the text tower's submodules are
named txtfusion.layerwise_blocks / txtfusion.refiner_blocks, so a merge loop that
filtered on a "blocks." prefix would drop 32 of 256 modules on a default LoRA and
still look like it worked.
"""

import types

import pytest
import torch

from src.pipelines.krea2 import SCHEDULE, Krea2Pipeline, _Krea2Sampler


TINY_CONFIG = dict(
    features=128,
    tdim=32,
    txtdim=64,
    heads=4,
    kvheads=2,
    multiplier=2,
    layers=2,
    patch=2,
    channels=16,
    txtheads=2,
    txtkvheads=2,
    txtlayers=12,
)


def _aitk():
    from src.pipelines.krea2 import _load_aitk_krea2_modules

    try:
        return _load_aitk_krea2_modules()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"ai-toolkit checkout not available: {e}")


@pytest.fixture(scope="module")
def tiny_transformer():
    mods = _aitk()
    torch.manual_seed(0)
    cfg = mods["mmdit"].SingleMMDiTConfig(**TINY_CONFIG)
    model = mods["mmdit"].SingleStreamDiT(cfg).to(torch.float32).eval()
    model.requires_grad_(False)
    return model


class _FakeVAE:
    """Stand-in for AutoencoderKLQwenImage: f8, 16 latent channels, video VAE."""

    class _Cfg:
        z_dim = 16
        latents_mean = [0.0] * 16
        latents_std = [1.0] * 16

    config = _Cfg()
    dtype = torch.float32

    def decode(self, latents):
        b, _, f, h, w = latents.shape
        return types.SimpleNamespace(sample=torch.zeros(b, 3, f, h * 8, w * 8))

    def encode(self, images):
        b, _, f, h, w = images.shape
        mean = torch.zeros(b, 16, f, h // 8, w // 8)
        return types.SimpleNamespace(
            latent_dist=types.SimpleNamespace(sample=lambda generator=None: mean)
        )


def _make_sampler(transformer, kv_cache=False):
    return _Krea2Sampler(
        transformer=transformer,
        vae=_FakeVAE(),
        device="cpu",
        dtype=torch.float32,
        patch_size=TINY_CONFIG["patch"],
        vae_scale_factor=8,
        kv_cache=kv_cache,
        schedule=dict(SCHEDULE),
    )


def _embeds(seq_len=7):
    return [torch.randn(seq_len, TINY_CONFIG["txtlayers"] * TINY_CONFIG["txtdim"])]


class TestKrea2Sampler:
    """The flow-matching loop, exercised against upstream's real predict_velocity."""

    def test_text_to_image_without_cfg(self, tiny_transformer):
        sampler = _make_sampler(tiny_transformer)
        out = sampler(
            conditional_embeds=_embeds(),
            height=128,
            width=128,
            num_inference_steps=3,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        assert len(out.images) == 1
        assert out.images[0].size == (128, 128)
        assert out.images[0].mode == "RGB"

    def test_cfg_path_runs_two_forwards(self, tiny_transformer):
        sampler = _make_sampler(tiny_transformer)
        out = sampler(
            conditional_embeds=_embeds(),
            unconditional_embeds=_embeds(),
            height=128,
            width=128,
            num_inference_steps=2,
            guidance_scale=3.0,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        assert out.images[0].size == (128, 128)

    def test_callback_fires_once_per_step(self, tiny_transformer):
        """ComfyUI progress and the cancel button ride on this callback -- it is
        the reason this repo owns the Euler loop instead of calling ai-toolkit's
        Krea2Pipeline, which has no callback parameter."""
        sampler = _make_sampler(tiny_transformer)
        seen = []

        def cb(pipe, i, t, kwargs):
            seen.append(i)
            return kwargs

        sampler(
            conditional_embeds=_embeds(),
            height=128,
            width=128,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(0),
            callback_on_step_end=cb,
        )
        assert seen == [0, 1, 2, 3]

    def test_same_seed_gives_same_output(self, tiny_transformer):
        sampler = _make_sampler(tiny_transformer)
        embeds = _embeds()

        def run(seed):
            return sampler(
                conditional_embeds=embeds,
                height=128,
                width=128,
                num_inference_steps=2,
                guidance_scale=0.0,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0].tobytes()

        assert run(7) == run(7)

    def test_reference_latents_are_accepted(self, tiny_transformer):
        """The edit path appends clean reference latents after the image tokens."""
        sampler = _make_sampler(tiny_transformer)
        refs = [[torch.randn(16, 4, 4), torch.randn(16, 6, 4)]]
        out = sampler(
            conditional_embeds=_embeds(),
            height=128,
            width=128,
            num_inference_steps=2,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(1),
            ref_latents=refs,
        )
        assert out.images[0].size == (128, 128)

    def test_kv_cache_path(self, tiny_transformer):
        """kv_cache implies isolate_refs; predict_velocity rejects one without
        the other, so this guards the pairing."""
        sampler = _make_sampler(tiny_transformer, kv_cache=True)
        refs = [[torch.randn(16, 4, 4)]]
        out = sampler(
            conditional_embeds=_embeds(),
            height=128,
            width=128,
            num_inference_steps=3,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(1),
            ref_latents=refs,
        )
        assert out.images[0].size == (128, 128)


class TestKrea2Schedule:
    def test_timestep_grid_runs_one_to_zero(self):
        p = _aitk()["pipeline"]
        ts = p.timesteps(4096, 30, 256, 6400, y1=0.5, y2=1.15, mu=None)
        assert len(ts) == 31
        assert ts[0] == pytest.approx(1.0)
        assert ts[-1] == pytest.approx(0.0)
        assert all(a > b for a, b in zip(ts, ts[1:])), "must decrease monotonically"

    def test_mu_is_resolution_dependent_when_unpinned(self):
        """SCHEDULE['mu'] is None so mu interpolates with token count, matching
        the trainer -- no krea2 preset pins model_kwargs.schedule_mu."""
        assert SCHEDULE["mu"] is None
        p = _aitk()["pipeline"]
        small = p.timesteps(256, 8, 256, 6400, y1=0.5, y2=1.15, mu=None)
        large = p.timesteps(6400, 8, 256, 6400, y1=0.5, y2=1.15, mu=None)
        assert small[1:-1] != large[1:-1], "mu must vary with resolution"


def _lora_target_modules(transformer):
    """Mirror ai-toolkit's gate: Linear modules whose name contains 'blocks'."""
    return [
        name
        for name, mod in transformer.named_modules()
        if isinstance(mod, torch.nn.Linear) and "blocks" in name
    ]


def _write_fake_lora(path, transformer, rank=4, seed=0):
    """Write a synthetic ai-toolkit-format krea2 LoRA.

    Matches what the trainer actually saves: 'diffusion_model.' prefix, peft
    lora_A/lora_B naming, and NO .alpha tensors (krea2 forces peft_format, which
    pins alpha == rank).
    """
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    sd = {}
    modules = dict(transformer.named_modules())
    for name in _lora_target_modules(transformer):
        lin = modules[name]
        out_f, in_f = lin.weight.shape
        sd[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(rank, in_f) * 0.01
        sd[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(out_f, rank) * 0.01
    save_file(sd, str(path))
    return sd


class TestKrea2LoraMerge:
    def test_merges_every_targeted_module(self, tiny_transformer, tmp_path, caplog):
        import copy

        transformer = copy.deepcopy(tiny_transformer)
        targets = _lora_target_modules(transformer)
        assert targets, "tiny model should have LoRA-targetable modules"

        lora_path = tmp_path / "krea2_lora.safetensors"
        _write_fake_lora(lora_path, transformer)

        pipe = Krea2Pipeline(device="cpu", offload_mode="none")
        before = {n: transformer.state_dict()[n + ".weight"].clone() for n in targets}

        with caplog.at_level("INFO"):
            pipe._merge_lora_to_transformer(transformer, str(lora_path), 1.0)

        after = transformer.state_dict()
        changed = [n for n in targets if not torch.equal(before[n], after[n + ".weight"])]
        assert len(changed) == len(targets), (
            f"only {len(changed)}/{len(targets)} modules changed -- "
            "the merge is dropping targets"
        )

    def test_includes_txtfusion_blocks(self, tiny_transformer):
        """A startswith('blocks.') filter would silently skip these."""
        targets = _lora_target_modules(tiny_transformer)
        txtfusion = [t for t in targets if t.startswith("txtfusion")]
        assert txtfusion, "txtfusion.*_blocks must be LoRA targets"
        assert all("blocks" in t for t in txtfusion)

    def test_merge_math_matches_scale(self, tiny_transformer, tmp_path):
        """W' == W + (alpha/rank) * scale * (B @ A); alpha defaults to rank."""
        import copy

        transformer = copy.deepcopy(tiny_transformer)
        rank = 4
        lora_path = tmp_path / "lora.safetensors"
        sd = _write_fake_lora(lora_path, transformer, rank=rank)

        name = _lora_target_modules(transformer)[0]
        w0 = transformer.state_dict()[name + ".weight"].clone()
        a = sd[f"diffusion_model.{name}.lora_A.weight"]
        b = sd[f"diffusion_model.{name}.lora_B.weight"]

        scale = 0.75
        pipe = Krea2Pipeline(device="cpu", offload_mode="none")
        pipe._merge_lora_to_transformer(transformer, str(lora_path), scale)

        expected = w0 + (b @ a) * scale  # alpha == rank -> internal scale 1.0
        got = transformer.state_dict()[name + ".weight"]
        assert torch.allclose(got, expected, atol=1e-5), (
            f"max diff {(got - expected).abs().max().item()}"
        )

    def test_scale_zero_is_a_noop(self, tiny_transformer, tmp_path):
        import copy

        transformer = copy.deepcopy(tiny_transformer)
        lora_path = tmp_path / "lora.safetensors"
        _write_fake_lora(lora_path, transformer)
        name = _lora_target_modules(transformer)[0]
        w0 = transformer.state_dict()[name + ".weight"].clone()

        pipe = Krea2Pipeline(device="cpu", offload_mode="none")
        pipe._merge_lora_to_transformer(transformer, str(lora_path), 0.0)

        assert torch.allclose(transformer.state_dict()[name + ".weight"], w0)

    def test_raises_when_nothing_merges(self, tiny_transformer, tmp_path):
        """A LoRA for a different architecture must fail loudly, not silently
        produce base-model output the user thinks is their LoRA."""
        from safetensors.torch import save_file

        lora_path = tmp_path / "wrong_arch.safetensors"
        save_file(
            {
                "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(4, 8),
                "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(8, 4),
            },
            str(lora_path),
        )

        pipe = Krea2Pipeline(device="cpu", offload_mode="none")
        with pytest.raises(ValueError, match="Merged 0 LoRA layers"):
            pipe._merge_lora_to_transformer(tiny_transformer, str(lora_path), 1.0)


class TestKrea2GenerateIntegration:
    """BasePipeline.generate() -> _run_inference -> sampler, including the
    ComfyUI progress/interrupt path that motivated owning the Euler loop."""

    @pytest.fixture
    def pipeline(self, tiny_transformer):
        pipe = Krea2Pipeline(device="cpu", offload_mode="none")
        pipe.pipe = _make_sampler(tiny_transformer)
        # Bypass the Qwen3-VL encoder: no weights are downloaded in these tests.
        pipe._encode_prompt = lambda prompt, vlm_images: torch.randn(
            7, TINY_CONFIG["txtlayers"] * TINY_CONFIG["txtdim"]
        )
        return pipe

    def test_generate_returns_image_and_seed(self, pipeline):
        out = pipeline.generate(
            prompt="x", width=128, height=128, num_inference_steps=2,
            guidance_scale=0.0, seed=1,
        )
        assert out["image"].size == (128, 128)
        assert out["seed"] == 1

    def test_step_observer_receives_progress(self, pipeline):
        from src.pipelines.base import pipeline_step_observer

        seen = []
        with pipeline_step_observer(lambda step, total, ts: seen.append((step, total))):
            pipeline.generate(
                prompt="x", width=128, height=128, num_inference_steps=4,
                guidance_scale=0.0, seed=1,
            )
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]

    def test_observer_can_interrupt(self, pipeline):
        """ComfyUI's cancel button raises from the observer."""
        from src.pipelines.base import pipeline_step_observer

        class Cancelled(Exception):
            pass

        def boom(step, total, ts):
            if step == 1:
                raise Cancelled()

        with pytest.raises(Cancelled):
            with pipeline_step_observer(boom):
                pipeline.generate(
                    prompt="x", width=128, height=128, num_inference_steps=5,
                    guidance_scale=0.0, seed=1,
                )

    def test_resolution_snaps_to_16(self, pipeline):
        out = pipeline.generate(
            prompt="x", width=130, height=127, num_inference_steps=2,
            guidance_scale=0.0, seed=1,
        )
        assert out["image"].size == (128, 112)

    def test_edit_pipeline_rejects_missing_control_image(self, tiny_transformer):
        from src.pipelines.krea2 import Krea2EditPipeline

        pipe = Krea2EditPipeline(device="cpu", offload_mode="none")
        pipe.pipe = _make_sampler(tiny_transformer, kv_cache=True)
        with pytest.raises(ValueError, match="requires at least one control image"):
            pipe.generate(
                prompt="x", width=128, height=128, num_inference_steps=2,
                guidance_scale=0.0, seed=1,
            )
