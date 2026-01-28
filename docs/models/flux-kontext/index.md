---
title: "FLUX Kontext (FLUX.1-Kontext-dev) LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run flux_kontext LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "flux_kontext lora inference, FLUX Kontext (FLUX.1-Kontext-dev) diffusers pipeline, ai-toolkit flux_kontext inference, training preview vs inference mismatch, flux-kontext"
permalink: /models/flux-kontext/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# FLUX Kontext (FLUX.1-Kontext-dev) LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `flux_kontext`  
**URL slug:** `flux-kontext`

This page documents the **reference Diffusers inference pipeline** for `flux_kontext` (FLUX Kontext (FLUX.1-Kontext-dev)). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/flux_kontext.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flux_kontext.py) |
| Base checkpoint | `black-forest-labs/FLUX.1-Kontext-dev` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | Required (`ctrl_img`) |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/flux_kontext.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/flux_kontext.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Control-image loading (base64/URL):** [`src/libs/image_utils.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/libs/image_utils.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "flux_kontext",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 25,
      "guidance_scale": 4.0,
      "neg": "",
      "ctrl_img": "<base64_or_url>"
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

### Control image

This model requires a control image. In the API request, set `ctrl_img` to either:

- a base64-encoded image, or
- an http(s) URL that the server can fetch.

## Pipeline behavior that matters

- Uses `diffusers.FluxControlPipeline` (control-image / editing variant).
- The server **always resizes** the control image to the snapped output size (BILINEAR).
- Disables internal auto-resize and pins `max_area=height*width` for the control conditioning path.

## Preview-matching notes (training preview vs inference mismatch)

- The most common mismatch is **control-image preprocessing**: this server uses BILINEAR resize to `(width, height)` after snapping.
- Width/height are floored to a multiple of **16**.
- If you try to reproduce in raw Diffusers, make sure you are using **FluxControlPipeline**, not FluxPipeline.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.
- If a control image is involved, match the **resize method** and the exact image(s) used.


## Related

- [FLUX.1-dev](../flux/)
