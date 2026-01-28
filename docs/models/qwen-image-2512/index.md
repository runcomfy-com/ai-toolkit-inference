---
title: "Qwen Image 2512 LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run qwen_image_2512 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "qwen_image_2512 lora inference, Qwen Image 2512 diffusers pipeline, ai-toolkit qwen_image_2512 inference, training preview vs inference mismatch, qwen-image-2512"
permalink: /models/qwen-image-2512/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Qwen Image 2512 LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `qwen_image_2512`  
**URL slug:** `qwen-image-2512`

This page documents the **reference Diffusers inference pipeline** for `qwen_image_2512` (Qwen Image 2512). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/qwen_image.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/qwen_image.py) |
| Base checkpoint | `Qwen/Qwen-Image-2512` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | No |
| LoRA scale behavior | Uses `fuse_lora()` (weights merged). Scale is fixed after load; changing `loras[].network_multiplier` triggers a reload. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/qwen_image.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/qwen_image.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "qwen_image_2512",
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

- Same inference graph as Qwen Image, but uses the **Qwen-Image-2512** base checkpoint.
- Guidance uses `true_cfg_scale`.
- LoRA is fused (`fuse_lora()` + `unload_lora_weights()`) to match the intended path for this variant.

## Preview-matching notes (training preview vs inference mismatch)

- Because LoRA is fused, the server cannot re-scale it between prompts without reload.
- Width/height are floored to a multiple of **32**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- [Qwen Image](../qwen-image/)
