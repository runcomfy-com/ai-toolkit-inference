---
title: "Qwen Image Edit Plus 2509 LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run qwen_image_edit_plus LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "qwen_image_edit_plus lora inference, Qwen Image Edit Plus 2509 diffusers pipeline, ai-toolkit qwen_image_edit_plus inference, training preview vs inference mismatch, qwen-image-edit-plus"
permalink: /models/qwen-image-edit-plus/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Qwen Image Edit Plus 2509 LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `qwen_image_edit_plus`  
**URL slug:** `qwen-image-edit-plus`

This page documents the **reference Diffusers inference pipeline** for `qwen_image_edit_plus` (Qwen Image Edit Plus 2509). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/qwen_image.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/qwen_image.py) |
| Base checkpoint | `Qwen/Qwen-Image-Edit-2509` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | Required (`ctrl_img_1..3`) |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/qwen_image.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/qwen_image.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Control-image loading (base64/URL):** [`src/libs/image_utils.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/libs/image_utils.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "qwen_image_edit_plus",
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
      "ctrl_img_1": "<base64_or_url>",
      "ctrl_img_2": "<optional>",
      "ctrl_img_3": "<optional>"
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

### Control images

This model expects **1–3 reference images**.

In this server implementation, provide them as:

- `ctrl_img_1` (required)
- `ctrl_img_2` (optional)
- `ctrl_img_3` (optional)

Notes:

- The **order** of images matters.
- For preview matching, keep the same preprocessing (size + resize kernel) as the reference pipeline.

## Pipeline behavior that matters

- Multi-image editing pipeline (`diffusers.QwenImageEditPlusPipeline`).
- The server expects **1–3 reference images** via `ctrl_img_1..3` (not `ctrl_img`).
- Prompt embeddings are encoded with preprocessed reference images using the pipeline’s `CONDITION_IMAGE_SIZE`.
- Guidance uses `true_cfg_scale`.

## Preview-matching notes (training preview vs inference mismatch)

- Provide reference images as `ctrl_img_1` (required), `ctrl_img_2`, `ctrl_img_3`. This implementation treats `ctrl_img_1..3` as the authoritative list.
- Changing the reference-image preprocessing (size/resize mode) is the fastest way to create preview mismatch.
- Width/height are floored to a multiple of **32**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.
- If a control image is involved, match the **resize method** and the exact image(s) used.


## Related

- [Qwen Image Edit Plus 2511](../qwen-image-edit-plus-2511/)
- [Qwen Image Edit](../qwen-image-edit/)
