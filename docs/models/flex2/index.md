---
title: "Flex.2-preview LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run flex2 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "flex2 lora inference, Flex.2-preview diffusers pipeline, ai-toolkit flex2 inference, training preview vs inference mismatch, flex2"
permalink: /models/flex2/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Flex.2-preview LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `flex2`  
**URL slug:** `flex2`

This page documents the **reference Diffusers inference pipeline** for `flex2` (Flex.2-preview). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/flex2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flex2.py) |
| Base checkpoint | `ostris/Flex.2-preview` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | No |
| LoRA scale behavior | Uses `fuse_lora()` (weights merged). Scale is fixed after load; changing `loras[].network_multiplier` triggers a reload. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/flex2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flex2.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "flex2",
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

- Instantiates an `AutoPipelineForText2Image` **custom pipeline** (`custom_pipeline="ostris/Flex.2-preview"`).
- Overrides the scheduler to `FlowMatchEulerDiscreteScheduler` with **shift=3.0** and **dynamic shifting enabled**.
- Uses `fuse_lora()` + `unload_lora_weights()` to match the intended inference path for this checkpoint.

## Preview-matching notes (training preview vs inference mismatch)

- Flex.2 is sensitive to scheduler wiring. If you run with a different scheduler (or shift), outputs will drift quickly.
- Because LoRA is fused, changing `loras[].network_multiplier` requires a pipeline reload.
- Width/height are floored to a multiple of **32**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.

## Related

- [Flex.1-alpha](../flex1/)
