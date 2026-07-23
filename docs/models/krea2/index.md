---
title: "Krea 2 LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run krea2 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "krea2 lora inference, Krea 2 diffusers pipeline, ai-toolkit krea2 inference, training preview vs inference mismatch, krea2"
permalink: /models/krea2/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Krea 2 LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `krea2`  
**URL slug:** `krea2`  
**AI Toolkit training arch:** `krea2`

This page documents the **reference inference pipeline** for `krea2` — the full (non-distilled) Krea 2 text-to-image checkpoint. It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth.

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview-vs-inference drift), run it via RunComfy's Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/krea2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/krea2.py) |
| Base checkpoint | `krea/Krea-2-Raw` (single file `raw.safetensors` at the repo root) |
| Text encoder | `Qwen/Qwen3-VL-4B-Instruct` (a stack of 12 hidden-state layers) |
| VAE | `Qwen/Qwen-Image` (`vae/` subfolder, f8, 16 latent channels) |
| Defaults | `sample_steps=30`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** (VAE f8 x patch 2) |
| Control image | No |
| LoRA scale behavior | **Merged into the transformer at load.** Changing `loras[].network_multiplier` triggers a full model reload. |
| Needs AI Toolkit | **Yes** — the transformer, packing and sampler come from `extensions_built_in/diffusion_models/krea2/`. |
| Gated weights | **Yes** — `krea/Krea-2-Raw` is HF-gated. `HF_TOKEN` must belong to an account that accepted the Krea 2 license. |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/krea2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/krea2.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Upstream model + sampler:** `extensions_built_in/diffusion_models/krea2/` in [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)

## Minimal API request

```json
{
  "model": "krea2",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 30,
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

- **`guidance_scale` is 0-normalized inside the model.** AI Toolkit computes
  `guidance = max(0, guidance_scale - 1)` before sampling, and this pipeline does exactly the
  same, so a `guidance_scale` here means the same thing as `sample.guidance_scale` in a training
  config. Do not subtract the 1 yourself.
- The flow-matching schedule shifts `mu` based on the image-token count (endpoints 256 → 0.5 and
  6400 → 1.15). `mu` is deliberately **not** pinned to 1.15 for the distilled checkpoints, because
  no AI Toolkit krea2 preset pins it either — preview parity wins over the vendor default.
- LoRA is merged directly into the transformer weights, so `network_multiplier` is fixed after
  load. A different scale forces a full reload (~26 GB checkpoint).
- LoRA keys are read in AI Toolkit's on-disk format (`diffusion_model.*`, peft-style
  `lora_A`/`lora_B`, no `.alpha` tensors — a missing alpha defaults to rank).

## Preview-matching notes (training preview vs inference mismatch)

- Start from **30 steps / guidance 4.0** — the AI Toolkit defaults for `krea2`.
- Width/height are floored to a multiple of **16**.
- The model needs roughly **33 GiB** of VRAM in bf16 (a ~12.8B transformer plus a 4B text
  encoder). This pipeline does not quantize, while AI Toolkit training defaults to float8; that
  is a known and deliberate parity delta.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps`, and remember `guidance_scale` is 0-normalized.
- Match `loras[].network_multiplier`.
- Confirm the LoRA was trained for `krea2` (safetensors metadata `ss_base_model_version` is `krea2`).

## Related

- [Krea 2 Turbo](../krea2-turbo/)
- [Krea 2 Edit](../krea2-o-edit/)
