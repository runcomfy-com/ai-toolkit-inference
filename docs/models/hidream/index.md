---
title: "HiDream-I1-Full LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run hidream LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "hidream lora inference, HiDream-I1-Full diffusers pipeline, ai-toolkit hidream inference, training preview vs inference mismatch, hidream"
permalink: /models/hidream/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# HiDream-I1-Full LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `hidream`  
**URL slug:** `hidream`

This page documents the **reference Diffusers inference pipeline** for `hidream` (HiDream-I1-Full). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/hidream.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/hidream.py) |
| Base checkpoint | `HiDream-ai/HiDream-I1-Full` |
| Defaults | `sample_steps=50`, `guidance_scale=5.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | No |
| LoRA scale behavior | Fused into transformer weights at load time; scale is fixed after load. |
| Needs AI Toolkit | Required (needs a local `ostris/ai-toolkit` checkout via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/hidream.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/hidream.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "hidream",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 50,
      "guidance_scale": 5.0,
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

- Uses AI Toolkit’s `HiDreamImagePipeline` and `HiDreamImageTransformer2DModel`.
- Loads **four** text encoders (CLIP×2 + T5 + Llama 3.1 8B) to match the model’s conditioning stack.
- Scheduler is `FlowUniPCMultistepScheduler` with shift=3.0.
- LoRA is fused directly into the transformer (AI Toolkit / ComfyUI `diffusion_model.` key format supported).

## Preview-matching notes (training preview vs inference mismatch)

- HiDream is extremely sensitive to its conditioning stack. If you use a different Llama checkpoint (or different max sequence length), outputs can drift.
- LoRA scale is fixed after fusion. Changing `loras[].network_multiplier` requires pipeline reload.
- Width/height are floored to a multiple of **16**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.


## Related

- See the [Model Catalog](../) for all supported models.
