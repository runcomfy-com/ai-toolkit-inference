---
title: "Wan 2.2 I2V A14B (14B) LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run wan22_14b_i2v LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "wan22_14b_i2v lora inference, Wan 2.2 I2V A14B (14B) diffusers pipeline, ai-toolkit wan22_14b_i2v inference, training preview vs inference mismatch, wan22-14b-i2v"
permalink: /models/wan22-14b-i2v/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Wan 2.2 I2V A14B (14B) LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `wan22_14b_i2v`  
**URL slug:** `wan22-14b-i2v`

This page documents the **reference Diffusers inference pipeline** for `wan22_14b_i2v` (Wan 2.2 I2V A14B (14B)). It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth (scheduler wiring, resolution snapping, LoRA application, and conditioning).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/wan22_i2v.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan22_i2v.py) |
| Base checkpoint | `ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** |
| Control image | Required (`ctrl_img`) |
| Video | Yes (`num_frames=41`, `fps=16` by default) |
| LoRA scale behavior | MoE LoRA (high/low noise) loaded into `transformer` + `transformer_2`. Scale is set per transformer via `loras[].network_multiplier`. |
| Needs AI Toolkit | Required (needs a local `ostris/ai-toolkit` checkout via `AI_TOOLKIT_PATH`) |

## Reference implementation (source of truth)

- **MoE LoRA validation (wan22 special-case):** [`src/api/v1/inference.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/api/v1/inference.py)
- **Pipeline implementation:** [`src/pipelines/wan22_i2v.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/wan22_i2v.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Control-image loading (base64/URL):** [`src/libs/image_utils.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/libs/image_utils.py)
- **Executor (prompt processing + caching):** [`src/tasks/executor.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/tasks/executor.py)

## Minimal API request

```json
{
  "model": "wan22_14b_i2v",
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
      "fps": 16,
      "ctrl_img": "<base64_or_url>"
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

### Control image

This model requires a control image. In the API request, set `ctrl_img` to either:

- a base64-encoded image, or
- an http(s) URL that the server can fetch.

## Pipeline behavior that matters

- Uses AI Toolkit’s custom `Wan22Pipeline` (not the stock diffusers I2V pipeline) for training-sample alignment.
- I2V-specific settings: `boundary_ratio=0.9`, `expand_timesteps=False`, `flow_shift=3.0`.
- Applies **first-frame conditioning** via `add_first_frame_conditioning`.
- LoRA uses MoE format (`loras` with `transformer: "low"` / `"high"`); `set_lora_scale()` allows changing scale between requests without full reload.

## Preview-matching notes (training preview vs inference mismatch)

- If you try to reproduce with `WanImageToVideoPipeline`, results will differ: this implementation uses the AI Toolkit **Wan22Pipeline** + first-frame conditioning.
- Resolution snapping is **16** for this I2V pipeline (different from Wan2.2 T2V’s 32).
- The control image is resized with **LANCZOS** and then encoded into the latent via the VAE for the first frame.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps` and the scheduler family (FlowMatch / UniPC / DDPM differences matter).
- Match `guidance_scale` semantics (some pipelines map it to a different internal parameter).
- Match `loras[].network_multiplier` and whether LoRA scale is dynamic vs fused.
- If a control image is involved, match the **resize method** and the exact image(s) used.
- If video: match `num_frames` and `fps` (and any frame-count constraints).


## Related

- [Wan 2.2 T2V A14B](../wan22-14b-t2v/)
- [Wan 2.1 I2V 14B](../wan21-i2v-14b/)
