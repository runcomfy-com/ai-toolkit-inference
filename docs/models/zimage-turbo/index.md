---
title: "Z-Image Turbo LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run zimage_turbo LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "zimage_turbo lora inference, Z-Image Turbo diffusers pipeline, ai-toolkit zimage_turbo inference, training preview vs inference mismatch, zimage-turbo"
permalink: /models/zimage-turbo/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Z-Image Turbo LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `zimage_turbo`  
**URL slug:** `zimage-turbo`

This page documents the **reference Diffusers inference pipeline** for `zimage_turbo` (Z-Image Turbo). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/zimage_turbo.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/zimage_turbo.py) |
| Base checkpoint | `Tongyi-MAI/Z-Image-Turbo` |
| Defaults | `sample_steps=8`, `guidance_scale=1.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | No |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/zimage_turbo.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/zimage_turbo.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "zimage_turbo",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 8,
      "guidance_scale": 1.0,
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

- Few-step model: defaults (8 steps / CFG 1.0) are part of the intended operating regime.
- Uses `diffusers.ZImagePipeline` if available; otherwise falls back to `FluxPipeline` (which changes results).
- LoRA is applied as an adapter and can be re-scaled per request.

## Preview-matching notes (training preview vs inference mismatch)

- Start from **8 steps / CFG 1.0**. Large changes in steps/CFG can look like ‘the LoRA isn’t applied’ for distilled few-step models.
- If your diffusers build doesn’t include `ZImagePipeline`, the fallback path is `FluxPipeline` (expect different outputs).
- Width/height are floored to a multiple of **32**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- [Z-Image De-Turbo](../zimage-deturbo/)
