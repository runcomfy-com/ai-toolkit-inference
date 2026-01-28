---
title: "OmniGen2 LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run omnigen2 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "omnigen2 lora inference, OmniGen2 diffusers pipeline, ai-toolkit omnigen2 inference, training preview vs inference mismatch, omnigen2"
permalink: /models/omnigen2/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# OmniGen2 LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `omnigen2`  
**URL slug:** `omnigen2`

This page documents the **reference Diffusers inference pipeline** for `omnigen2` (OmniGen2). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/omnigen2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/omnigen2.py) |
| Base checkpoint | `OmniGen2/OmniGen2` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | Optional (`ctrl_img` or `ctrl_img_1..3`) |
| LoRA scale behavior | Fused into transformer weights at load time; scale is fixed after load. |
| Needs AI Toolkit | Required (needs a local `ostris/ai-toolkit` checkout via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/omnigen2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/omnigen2.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "omnigen2",
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

- Multi-modal pipeline: uses **Qwen2.5-VL** as the MLLM + a custom `OmniGen2Transformer2DModel`.
- Supports image editing via `ctrl_img` (single) or `ctrl_img_1..3` (multiple reference images).
- Applies the model’s **chat template** before calling the pipeline (to match training behavior).
- Uses `text_guidance_scale` (mapped from the API’s `guidance_scale`) and keeps `image_guidance_scale=1.0`.
- LoRA is fused into transformer weights (AI Toolkit / ComfyUI `diffusion_model.` key format supported).

## Preview-matching notes (training preview vs inference mismatch)

- If your prompts are not wrapped in the same **chat template**, results can look like ‘the LoRA isn’t working’ even when it is.
- LoRA scale is fixed after fusion. Changing `loras[].network_multiplier` requires pipeline reload.
- If you provide multiple reference images, order matters. The server accepts up to 3 (`ctrl_img_1..3`).
- Width/height are floored to a multiple of **16**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- [Qwen Image Edit Plus](../qwen-image-edit-plus/)
- [HiDream](../hidream/)
