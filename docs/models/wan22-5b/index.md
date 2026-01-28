---
title: "Wan 2.2 TI2V 5B LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run wan22_5b LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "wan22_5b lora inference, Wan 2.2 TI2V 5B diffusers pipeline, ai-toolkit wan22_5b inference, training preview vs inference mismatch, wan22-5b"
permalink: /models/wan22-5b/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Wan 2.2 TI2V 5B LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `wan22_5b`  
**URL slug:** `wan22-5b`

This page documents the **reference Diffusers inference pipeline** for `wan22_5b` (Wan 2.2 TI2V 5B). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/wan22_5b.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan22_5b.py) |
| Base checkpoint | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | Optional (`ctrl_img` enables I2V / first-frame conditioning) |
| Video | Yes (`num_frames=41`, `fps=16` by default) |
| LoRA scale behavior | Dynamic via adapters (`set_adapters`). Scale is set per request via `loras[].network_multiplier`. |
| Needs AI Toolkit | Required (needs a local `ostris/ai-toolkit` checkout via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/wan22_5b.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan22_5b.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "wan22_5b",
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

- Unified model id for both modes: with `ctrl_img` → I2V (first-frame conditioning); without → T2V.
- Uses AI Toolkit’s `Wan22Pipeline` with `expand_timesteps=True` and `flow_shift=5.0` (5B-specific).
- In I2V mode, computes both **conditioned latents** and a **noise_mask** (AI Toolkit v2.2 conditioning helper).

## Preview-matching notes (training preview vs inference mismatch)

- The server forces `num_frames` into a **4N+1** pattern (e.g., 41). If you compare with a run that used a different frame count, the motion profile will differ.
- I2V mode requires VAE-on-GPU to encode the first frame; this is why CPU offload is disabled in this pipeline implementation.
- Width/height are floored to a multiple of **32**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.
- If video: match `num_frames` and `fps` (and any frame-count constraints).


## Related

- [Wan 2.2 T2V 14B](../wan22-14b-t2v/)
- [LTX-2](../ltx2/)
