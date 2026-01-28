---
title: "Stable Diffusion 1.5 LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run sd15 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "sd15 lora inference, Stable Diffusion 1.5 diffusers pipeline, ai-toolkit sd15 inference, training preview vs inference mismatch, sd15"
permalink: /models/sd15/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Stable Diffusion 1.5 LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `sd15`  
**URL slug:** `sd15`

This page documents the **reference Diffusers inference pipeline** for `sd15` (Stable Diffusion 1.5). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/sd15.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/sd15.py) |
| Base checkpoint | `runwayml/stable-diffusion-v1-5` |
| Defaults | `sample_steps=25`, `guidance_scale=6.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **8** |
| Control image | No |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/sd15.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/sd15.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "sd15",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 25,
      "guidance_scale": 6.0,
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

- Uses `diffusers.StableDiffusionPipeline`.
- Pins a `DDPMScheduler` configuration (to match the intended training/inference defaults).
- LoRA is applied as an adapter and can be re-scaled per request.

## Preview-matching notes (training preview vs inference mismatch)

- Scheduler configuration differences (DDPM vs Euler, etc.) are the most common cause of SD1.5 drift when trying to match sample previews.
- Width/height are floored to a multiple of **8**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- [SDXL](../sdxl/)
