---
title: "LTX-2 LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run ltx2 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "ltx2 lora inference, LTX-2 diffusers pipeline, ai-toolkit ltx2 inference, training preview vs inference mismatch, ltx2"
permalink: /models/ltx2/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# LTX-2 LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `ltx2`  
**URL slug:** `ltx2`

This page documents the **reference Diffusers inference pipeline** for `ltx2` (LTX-2). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/ltx2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/ltx2.py) |
| Base checkpoint | `Lightricks/LTX-2` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | Optional (`ctrl_img` switches to I2V) |
| Video | Yes (`num_frames=41`, `fps=24` by default) |
| LoRA scale behavior | LoRA is converted (AI Toolkit → diffusers) and then applied via `fuse_lora()`; scale is fixed after load. |
| Needs AI Toolkit | Optional (recommended for LoRA conversion helpers via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/ltx2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/ltx2.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "ltx2",
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
      "fps": 24
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

- Supports both **text-to-video** and **image-to-video**: providing `ctrl_img` switches to `LTX2ImageToVideoPipeline`.
- Returns **video frames + audio** (the executor writes an MP4).
- LoRA loading path converts AI Toolkit / original-format weights into diffusers keys, then uses `fuse_lora()`.
- Uses `output_type="np"` to match AI Toolkit-style postprocessing (uint8 frames).

## Preview-matching notes (training preview vs inference mismatch)

- LTX-2 requires `num_frames` in an **8N+1** pattern (1, 9, 17, 25, 33, 41...). The diffusers pipeline may adjust internally; mismatched frame counts are a common source of ‘it looks different’.
- LoRA is fused, so scale changes require reload.
- For I2V, the server resizes the control image to `(width, height)` using **LANCZOS**.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.
- If video: match `num_frames` and `fps` (and any frame-count constraints).


## Related

- [Wan 2.2 T2V 14B](../wan22-14b-t2v/)
- [Wan 2.2 I2V 14B](../wan22-14b-i2v/)
