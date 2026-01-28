---
title: "Wan 2.1 T2V 14B LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run wan21_14b LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "wan21_14b lora inference, Wan 2.1 T2V 14B diffusers pipeline, ai-toolkit wan21_14b inference, training preview vs inference mismatch, wan21-14b"
permalink: /models/wan21-14b/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Wan 2.1 T2V 14B LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `wan21_14b`  
**URL slug:** `wan21-14b`

This page documents the **reference Diffusers inference pipeline** for `wan21_14b` (Wan 2.1 T2V 14B). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/wan21.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan21.py) |
| Base checkpoint | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | No |
| Video | Yes (`num_frames=41`, `fps=16` by default) |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/wan21.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan21.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "wan21_14b",
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
      "num_frames": 41,
      "fps": 16
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

- Uses `diffusers.WanPipeline` for text-to-video.
- LoRA is loaded via `load_lora_weights` + `set_adapters`; scale is set per request via `loras[].network_multiplier`.

## Preview-matching notes (training preview vs inference mismatch)

- Video models add two extra degrees of freedom: `num_frames` and `fps`. Match them when comparing outputs.
- Width/height are floored to a multiple of **16**.
- The server’s default seeding behavior increments the seed per prompt (42, 43, 44...). For strict reproducibility, set `seed` explicitly per prompt.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.
- If video: match `num_frames` and `fps` (and any frame-count constraints).


## Related

- [Wan 2.1 T2V 1.3B](../wan21-1b/)
- [Wan 2.1 I2V 14B](../wan21-i2v-14b/)
- [Wan 2.2 T2V 14B](../wan22-14b-t2v/)
