from PIL import Image


def test_rcflux2_node_passes_safe_prompt_and_negative_prompt(monkeypatch):
    """
    The RCFlux2 ComfyUI node should never pass None into the vendor pipeline.
    It should also pass list-wrapped prompts to match ai-toolkit FLUX.2 behavior
    (some versions call len(prompt) directly).
    """
    from comfyui_nodes.rc_models import RCFlux2

    class _VendorResult:
        def __init__(self):
            self.images = [Image.new("RGB", (8, 8), color=(0, 0, 0))]

    class _VendorPipe:
        def __call__(self, **kwargs):
            assert kwargs["prompt"] == ["hello"]
            assert kwargs["negative_prompt"] == [""]
            assert kwargs.get("control_img_list") is None
            return _VendorResult()

    class _DummyFlux2Pipeline:
        # RCFlux2 uses pipe.pipe as the vendor callable.
        pipe = _VendorPipe()

    # Patch get_or_load_pipeline in the module under test.
    import comfyui_nodes.rc_models as rc_models

    monkeypatch.setattr(rc_models, "get_or_load_pipeline", lambda **_kwargs: _DummyFlux2Pipeline())

    node = RCFlux2()
    # Avoid importing the real Flux2Pipeline (which pulls in diffusers) in this unit test.
    monkeypatch.setattr(RCFlux2, "_pipeline_ctor", lambda self: (lambda *a, **k: None))
    (img_tensor,) = node.generate(
        prompt="hello",
        width=64,
        height=64,
        sample_steps=1,
        guidance_scale=1.0,
        seed=42,
        lora_path="",
        lora_scale=1.0,
        hf_token="",
        negative_prompt="",
    )

    # ComfyUI IMAGE tensor: [B,H,W,C]
    assert tuple(img_tensor.shape) == (1, 8, 8, 3)
