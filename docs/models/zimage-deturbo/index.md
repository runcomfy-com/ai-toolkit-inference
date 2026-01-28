---
title: "Z-Image De-Turbo LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run zimage_deturbo LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "zimage_deturbo lora inference, Z-Image De-Turbo diffusers pipeline, ai-toolkit zimage_deturbo inference, training preview vs inference mismatch, zimage-deturbo"
permalink: /models/zimage-deturbo/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Z-Image De-Turbo LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `zimage_deturbo`  
**URL slug:** `zimage-deturbo`

This page documents the **reference Diffusers inference pipeline** for `zimage_deturbo` (Z-Image De-Turbo). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/zimage_deturbo.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/zimage_deturbo.py) |
| Base checkpoint | `Tongyi-MAI/Z-Image-Turbo (base) + ostris/Z-Image-De-Turbo (transformer)` |
| Defaults | `sample_steps=25`, `guidance_scale=3.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | No |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/zimage_deturbo.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/zimage_deturbo.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "zimage_deturbo",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 25,
      "guidance_scale": 3.0,
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

- Builds a Z-Image pipeline by mixing components: base from **Z-Image-Turbo**, but transformer from **Z-Image-De-Turbo**.
- Uses a `FlowMatchEulerDiscreteScheduler` configured with `shift=3.0` (no dynamic shifting).
- Loads Qwen3 text encoder + tokenizer directly and constructs the pipeline manually (avoids meta-tensor issues).

## Preview-matching notes (training preview vs inference mismatch)

- De-Turbo is not ‘Turbo with more steps’ — it uses a different transformer checkpoint. If you load the wrong transformer, you will not match this server.
- Width/height are floored to a multiple of **32**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- [Z-Image](../zimage/)
- [Z-Image Turbo](../zimage-turbo/)
