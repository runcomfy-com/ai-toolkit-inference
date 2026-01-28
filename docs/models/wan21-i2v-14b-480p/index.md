---
title: "Wan 2.1 I2V 14B (480P) LoRA Inference with Diffusers"
description: "Run Wan 2.1 I2V 14B 480P LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference Diffusers pipeline. Defaults, control-image behavior, and preview-matching pitfalls."
keywords: "wan 2.1 i2v 480p lora inference, wan21_i2v_14b480p, wan2.1 image-to-video diffusers, WanImageToVideoPipeline, ai-toolkit wan inference, training preview mismatch"
permalink: /models/wan21-i2v-14b-480p/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Wan 2.1 I2V 14B (480P) LoRA Inference with Diffusers (AI Toolkit-trained)

This page documents **Wan 2.1 image-to-video (I2V)** LoRA inference for the **480P** checkpoint variant, using the **ai-toolkit-inference** reference pipeline.

If you trained a LoRA with `ostris/ai-toolkit` and your **inference doesn’t match training samples**, the biggest lever for Wan I2V is usually: **using the correct base checkpoint (480P vs 720P), matching the requested resolution, and matching control-image preprocessing**.

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Reference implementation (source of truth)

- **API model id:** `wan21_i2v_14b480p`
- **Pipeline implementation:** [`src/pipelines/wan21.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan21.py)
- **Base pipeline behaviors (seed, resolution snapping, LoRA loading):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names you pass):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)

Related model page:
- [Wan 2.1 I2V 14B (720P)](../wan21-i2v-14b/)

## Defaults used by the server

| Setting | Value | Where it comes from |
|---|---:|---|
| Base checkpoint | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | `Wan21I2V14B480PPipeline.CONFIG` in `wan21.py` |
| Resolution divisor (snapping) | `16` | `PipelineConfig.resolution_divisor` default in `base.py` |
| Default steps | `25` | `PipelineConfig.default_steps` default in `base.py` |
| Default guidance scale | `4.0` | `PipelineConfig.default_guidance_scale` default in `base.py` |
| Default frames / fps | `41` frames, `16` fps | `PipelineConfig.default_num_frames/default_fps` defaults in `base.py` |
| Control image required | Yes (`ctrl_img`) | `Wan21I2V14BPipeline.CONFIG.requires_control_image=True` in `wan21.py` |

## What makes this pipeline “480P”

The **only difference** between `wan21_i2v_14b` and `wan21_i2v_14b480p` in this repo is the **base model checkpoint**:

- `wan21_i2v_14b` → `...-720P-Diffusers`
- `wan21_i2v_14b480p` → `...-480P-Diffusers`

The inference logic is otherwise the same (same scheduler configuration, same control-image resize behavior, same LoRA application path).

Practical implication:
- This checkpoint is tuned for **480p-ish outputs**. For best stability and speed, use dimensions that look like 480p **and are divisible by 16** (e.g. `832×480`, `848×480`, `864×480`).

## Control image behavior (I2V)

This model **requires** a control image:

- Request field: `ctrl_img` (base64 or URL, see the API docs)
- The server **resizes** the control image to exactly match `width×height` before passing it into Diffusers.

Why this matters for preview matching:
- If your training samples used a different resize rule / crop policy / aspect bucket, the output can drift even with the same seed.

## LoRA loading and scale

This pipeline uses the default Diffusers LoRA path:

- `pipe.load_lora_weights(...)`
- `pipe.set_adapters(..., adapter_weights=[network_multiplier])`

So **LoRA scale is dynamically adjustable** per request (no permanent merge like `fuse_lora`).

## Minimal HTTP request example

```json
{
  "model": "wan21_i2v_14b480p",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a cinematic shot of a runner on a beach, handheld camera",
      "neg": "",
      "seed": 42,
      "width": 832,
      "height": 480,
      "num_frames": 41,
      "fps": 16,
      "sample_steps": 25,
      "guidance_scale": 4.0,
      "ctrl_img": "https://example.com/first_frame.png"
    }
  ],
  "loras": [
    {
      "path": "my_wan_job/my_wan_job.safetensors",
      "network_multiplier": 1.0
    }
  ]
}
```

Notes:
- The server snaps `width/height` **down** to a multiple of 16.
- If you omit `seed`, the server uses the pipeline default seed (usually `42`) **plus the prompt index** (0-based).
- To force randomization, set `seed: -1` (the server generates a random seed and returns it in the outputs).

## Preview vs inference mismatch (Wan I2V 80/20)

The most common reasons outputs drift from AI Toolkit samples:

1) **Wrong checkpoint variant (480P vs 720P)**
- If samples were produced with the 480P model but you infer with the 720P checkpoint (or vice versa), the look and motion can shift.

2) **Different resolution after snapping**
- The server floors width/height to a multiple of 16. Even a small change in aspect can materially change I2V motion.

3) **Control image preprocessing mismatch**
- This pipeline resizes `ctrl_img` to exactly `width×height`. If another stack crops/letterboxes instead, you’ll see drift.


## Related

- [Wan 2.1 I2V 14B (720P)](../wan21-i2v-14b/)
- [Wan 2.1 T2V 14B](../wan21-14b/)
