import pytest
from PIL import Image


def test_flux2_pipeline_always_passes_non_none_negative_prompt():
    """
    Regression test for ComfyUI crash:
    TypeError: object of type 'NoneType' has no len()

    The ai-toolkit FLUX.2 vendor pipeline may call len(negative_prompt), so our
    wrapper must never pass None (even if we ignore negative prompts).
    """
    pytest.importorskip("diffusers")
    from src.pipelines.flux2 import Flux2Pipeline

    class _DummyResult:
        def __init__(self):
            self.images = [Image.new("RGB", (8, 8), color=(0, 0, 0))]

    class _DummyPipe:
        def __call__(self, **kwargs):
            # The entire point: never let this be None.
            assert "negative_prompt" in kwargs
            assert kwargs["negative_prompt"] is not None
            return _DummyResult()

    p = Flux2Pipeline(device="cpu", offload_mode="none", hf_token=None)
    p.pipe = _DummyPipe()

    out = p._run_inference(
        prompt="hello",
        negative_prompt="",
        width=64,
        height=64,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=None,  # not used by our dummy pipe
        control_image=None,
        control_images=None,
        num_frames=1,
        fps=16,
    )

    assert "image" in out
