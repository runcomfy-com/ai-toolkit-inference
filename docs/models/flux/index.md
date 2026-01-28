---
title: "FLUX.1-dev LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run flux LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "flux lora inference, FLUX.1-dev diffusers pipeline, ai-toolkit flux inference, training preview vs inference mismatch, flux"
permalink: /models/flux/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# FLUX.1-dev LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `flux`  
**URL slug:** `flux`

This page documents the **reference Diffusers inference pipeline** for `flux` (FLUX.1-dev). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/flux_dev.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flux_dev.py) |
| Base checkpoint | `black-forest-labs/FLUX.1-dev` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | No |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/flux_dev.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flux_dev.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "flux",
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

- Reference implementation uses `diffusers.FluxPipeline` (`FluxPipeline.from_pretrained(...)`).
- LoRA is loaded as an adapter; scale is set per request via `loras[].network_multiplier` and can be updated without reload.

## Preview-matching notes (training preview vs inference mismatch)

- Width/height are floored to a multiple of **16**.
- If you are comparing with a UI that uses a different FLUX checkpoint (e.g., schnell vs dev), you will not get matching outputs even with the same LoRA.
- The API uses a single LoRA scale per request via `loras[].network_multiplier`; send separate requests if you need multiple scales.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- [FLUX Kontext](../flux-kontext/)
- [FLUX.2-dev](../flux2/)
