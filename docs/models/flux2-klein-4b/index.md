---
title: "FLUX.2-klein 4B LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run flux2_klein_4b LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "flux2_klein_4b lora inference, FLUX.2-klein 4B diffusers pipeline, ai-toolkit flux2 klein inference, training preview vs inference mismatch, flux2-klein-4b"
permalink: /models/flux2-klein-4b/
---

[Docs Home](../../) | [Model Catalog](../) | [HTTP API](../../api/) | [Troubleshooting](../../troubleshooting/)
# FLUX.2-klein 4B LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `flux2_klein_4b`  
**URL slug:** `flux2-klein-4b`

This page documents the **reference Diffusers inference pipeline** for `flux2_klein_4b` (FLUX.2-klein 4B). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview-vs-inference drift), run it via RunComfy's Cloud AI Toolkit (Train + Inference). You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/flux2_klein.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flux2_klein.py) |
| Base checkpoint | `black-forest-labs/FLUX.2-klein-base-4B` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | No |
| LoRA scale behavior | Manual LoRA merge into the transformer at load time; scale is fixed after load. |
| Needs AI Toolkit | Required (needs a local `ostris/ai-toolkit` checkout via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/flux2_klein.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flux2_klein.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model -> class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "flux2_klein_4b",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 25,
      "guidance_scale": 4.0,
      "neg": ""
    }
  ],
  "loras": [
    {
      "path": "my_lora_job/my_lora.safetensors",
      "network_multiplier": 1.0
    }
  ]
}
```

## Pipeline behavior that matters

- Uses AI Toolkit's **custom FLUX.2 pipeline + transformer** with **Klein 4B params** to match training sample previews.
- Loads a **Qwen3** text encoder and uses the Qwen prompt-encoding path in the pipeline.
- Uses the shared FLUX.2 VAE weights from **`ai-toolkit/flux2_vae`**.
- Not guidance-distilled: **negative_prompt is used** when `guidance_scale > 1.0`.
- Scheduler is `CustomFlowMatchEulerDiscreteScheduler` (AI Toolkit sampler).
- LoRA is expected in **ComfyUI / AI Toolkit format** (`diffusion_model.` prefix) and is merged into transformer weights.

## Preview-matching notes (training preview vs inference mismatch)

- LoRA is merged into weights; changing `loras[].network_multiplier` requires a pipeline reload.
- If you run the official diffusers FLUX.2 pipeline without the AI Toolkit transformer/pipeline wiring, results may differ from AI Toolkit sample previews.
- Width/height are floored to a multiple of **16**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` and `neg` (CFG is active on Klein models).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.

## Related

- [FLUX.2-dev](../flux2/)
- [FLUX.2-klein 9B](../flux2-klein-9b/)
