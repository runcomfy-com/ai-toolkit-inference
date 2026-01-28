---
title: "HiDream-E1-Full LoRA Image Editing with Diffusers (AI Toolkit-trained)"
description: "Run hidream_e1 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference editing pipeline. Defaults, required control image input, and preview vs inference mismatch notes."
keywords: "hidream e1 lora inference, hidream_e1 diffusers pipeline, ai-toolkit hidream e1 editing, control image, training preview vs inference mismatch"
permalink: /models/hidream-e1/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# HiDream-E1-Full LoRA Image Editing with Diffusers (AI Toolkit-trained)

**API model id:** `hidream_e1`  
**URL slug:** `hidream-e1`

This page documents the **reference Diffusers editing pipeline** for `hidream_e1` (HiDream-E1-Full). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/hidream.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/hidream.py) (class `HiDreamE1Pipeline`) |
| Base checkpoint | `HiDream-ai/HiDream-E1-Full` |
| Defaults | `sample_steps=28`, `guidance_scale=5.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | **Required** (`ctrl_img`) |
| LoRA scale behavior | Fused into transformer weights at load time; changing `loras[].network_multiplier` requires reload. |
| Needs AI Toolkit | Required (needs a local `ostris/ai-toolkit` checkout via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/hidream.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/hidream.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **API routes and validation:** [`src/api/v1/inference.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/api/v1/inference.py)

## Minimal API request

```json
{
  "model": "hidream_e1",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] make it look like a watercolor illustration",
      "neg": "",
      "seed": 42,
      "width": 1024,
      "height": 1024,
      "sample_steps": 28,
      "guidance_scale": 5.0,
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

## Pipeline behavior that matters

- `hidream_e1` is an **editing** pipeline: `ctrl_img` is mandatory and is resized to the requested output size.
- The LoRA is **fused** into the transformer at load time (no dynamic adapter scaling).
- The pipeline uses an internal `image_guidance_scale` (currently fixed in code) in addition to `guidance_scale`.

## Preview-matching notes (training preview vs inference mismatch)

- HiDream is sensitive to its conditioning stack (tokenizers/models/max sequence length). If you use a different stack than the reference pipeline, outputs can drift.
- Width/height are floored to a multiple of **16**.
- Because LoRA is fused, changing `loras[].network_multiplier` requires a pipeline reload.


## Related

- [HiDream I1 (text-to-image)](../hidream/)
- See the [Model Catalog](../) for all supported models.
