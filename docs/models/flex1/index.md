---
title: "Flex.1-alpha LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run flex1 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "flex1 lora inference, Flex.1-alpha diffusers pipeline, ai-toolkit flex1 inference, training preview vs inference mismatch, flex1"
permalink: /models/flex1/
---

← [Docs Home](../../) · [Model Catalog](../)

# Flex.1-alpha LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `flex1`  
**URL slug:** `flex1`

This page documents the **reference Diffusers inference pipeline** for `flex1` (Flex.1-alpha). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/flex1_alpha.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flex1_alpha.py) |
| Base checkpoint | `ostris/Flex.1-alpha` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | No |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/flex1_alpha.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flex1_alpha.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "flex1",
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

- Uses `diffusers.FluxPipeline` under the hood (Flex.1-alpha is FLUX-family).
- LoRA is applied as a diffusers adapter (`load_lora_weights` + `set_adapters`).

## Preview-matching notes (training preview vs inference mismatch)

- Flex.1-alpha snaps width/height to a multiple of **16**. If your UI rounds instead of floors, the effective resolution can differ.
- If you compare against Flex.2-preview or other FlowMatch-tuned pipelines, make sure you are actually using the **Flex.1-alpha / FluxPipeline graph**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.

## Common search intents

People land on this page searching for things like:

- Flex.1-alpha LoRA inference
- ostris/Flex.1-alpha diffusers FluxPipeline
- Flex1 LoRA set_adapters
- ai-toolkit Flex.1 alpha inference
- Flex.1 preview mismatch

## Related

- [Flex.2-preview](../flex2/)
