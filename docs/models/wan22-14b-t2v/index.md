---
title: "Wan 2.2 T2V A14B (14B) LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run wan22_14b_t2v LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "wan22_14b_t2v lora inference, Wan 2.2 T2V A14B (14B) diffusers pipeline, ai-toolkit wan22_14b_t2v inference, training preview vs inference mismatch, wan22-14b-t2v"
permalink: /models/wan22-14b-t2v/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Wan 2.2 T2V A14B (14B) LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `wan22_14b_t2v`  
**URL slug:** `wan22-14b-t2v`

This page documents the **reference Diffusers inference pipeline** for `wan22_14b_t2v` (Wan 2.2 T2V A14B (14B)). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/wan22_t2v.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan22_t2v.py) |
| Base checkpoint | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | No |
| Video | Yes (`num_frames=41`, `fps=16` by default) |
| LoRA scale behavior | MoE LoRA (high/low noise) loaded into `transformer` + `transformer_2`. Scale is set per transformer via `loras[].network_multiplier`. |
| Needs AI Toolkit | No |

## Reference implementation (source of truth)

- **MoE LoRA validation (wan22 special-case):** [`src/api/v1/inference.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/api/v1/inference.py)
- **Pipeline implementation:** [`src/pipelines/wan22_t2v.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan22_t2v.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "wan22_14b_t2v",
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
      "path": "my_lora_job/my_high_noise.safetensors",
      "transformer": "high",
      "network_multiplier": 1.0
    },
    {
      "path": "my_lora_job/my_low_noise.safetensors",
      "transformer": "low",
      "network_multiplier": 1.0
    }
  ]
}
```



## Pipeline behavior that matters

- Uses `diffusers.WanPipeline` but swaps in a transformer from **`ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16`** to better match AI Toolkit training samples.
- LoRA is **MoE format**: `loras` with `transformer: "low"` / `"high"`, loaded into two transformer stacks.
- Implements `set_lora_scale()` to update both adapters without reload.

## Preview-matching notes (training preview vs inference mismatch)

- If you load only one side of the MoE LoRA (high or low), the LoRA effect will not match AI Toolkit samples.
- This model snaps resolution to a multiple of **32**.
- LoRA scale is set via `loras[].network_multiplier`; this pipeline exposes `set_lora_scale()` so changing scale between requests doesn’t require a full reload.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.
- If video: match `num_frames` and `fps` (and any frame-count constraints).


## Related

- [Wan 2.2 I2V A14B](../wan22-14b-i2v/)
- [Wan 2.2 TI2V 5B](../wan22-5b/)
