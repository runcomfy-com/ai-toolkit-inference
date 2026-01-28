---
title: "Chroma1 Base LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run chroma LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "chroma lora inference, Chroma1 Base diffusers pipeline, ai-toolkit chroma inference, training preview vs inference mismatch, chroma"
permalink: /models/chroma/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Chroma1 Base LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `chroma`  
**URL slug:** `chroma`

This page documents the **reference Diffusers inference pipeline** for `chroma` (Chroma1 Base). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/chroma.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/chroma.py) |
| Base checkpoint | `lodestones/Chroma1-Base` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | No |
| LoRA scale behavior | Merged into the transformer **before float8 quantization**. Scale is fixed after load. |
| Needs AI Toolkit | Required (needs a local `ostris/ai-toolkit` checkout via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/chroma.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/chroma.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "chroma",
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

- Loads a custom **Chroma** transformer from AI Toolkit and builds an `AITK ChromaPipeline` wrapper.
- Quantizes the transformer and T5 text encoder to **float8** (AI Toolkit `torchao` path).
- Uses a **CustomFlowMatchEulerDiscreteScheduler** with dynamic shifting (AI Toolkit sampler).
- Prompt embeddings are computed via **T5**; the `negative_prompt` string is not used (unconditional embeds are always empty-string).

## Preview-matching notes (training preview vs inference mismatch)

- **LoRA scale is baked in**: the server merges LoRA weights into the transformer and then quantizes. If you change `loras[].network_multiplier`, the executor reloads the pipeline for that scale.
- Quantization is part of the reference behavior. If you run Chroma in full precision elsewhere, results can shift even with identical steps/seed.
- The server always uses an empty-string unconditional embedding for the negative path; passing a non-empty `neg` will not change the unconditional embedding.
- Width/height are floored to a multiple of **32** (resolution bucket divisibility).

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- See the [Model Catalog](../) for all supported models.
