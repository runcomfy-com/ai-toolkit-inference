---
title: "Krea 2 Edit Turbo LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run krea2_o_edit_turbo LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "krea2_o_edit_turbo lora inference, Krea 2 Edit Turbo diffusers pipeline, ai-toolkit krea2_o_edit_turbo inference, training preview vs inference mismatch, krea2-o-edit-turbo"
permalink: /models/krea2-o-edit-turbo/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Krea 2 Edit Turbo LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `krea2_o_edit_turbo`  
**URL slug:** `krea2-o-edit-turbo`  
**AI Toolkit training arch:** `krea2:o_edit_turbo`

This page documents the **reference inference pipeline** for `krea2_o_edit_turbo` — Krea 2 Turbo with in-context reference-image editing. It is designed for running **LoRAs trained with** `ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth.

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview-vs-inference drift), run it via RunComfy's Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/krea2.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/krea2.py) |
| Base checkpoint | `krea/Krea-2-Turbo` (single file `turbo.safetensors` at the repo root) |
| Text encoder | `Qwen/Qwen3-VL-4B-Instruct` (a stack of 12 hidden-state layers) |
| VAE | `Qwen/Qwen-Image` (`vae/` subfolder, f8, 16 latent channels) |
| Defaults | `sample_steps=8`, `guidance_scale=1.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **16** (VAE f8 x patch 2) |
| Control image | Yes — 1–3 reference images via `ctrl_img` / `ctrl_img_1..3`. **Required.** |
| LoRA scale behavior | **Merged into the transformer at load.** Changing `loras[].network_multiplier` triggers a full model reload. |
| Needs AI Toolkit | **Yes** — the transformer, packing and sampler come from `extensions_built_in/diffusion_models/krea2/`. |
| Gated weights | **Yes** — `krea/Krea-2-Turbo` is HF-gated. `HF_TOKEN` must belong to an account that accepted the Krea 2 license. |

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
  "model": "krea2_o_edit_turbo",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 1024,
      "height": 1024,
      "seed": 42,
      "sample_steps": 8,
      "guidance_scale": 1.0,
      "neg": "",
      "ctrl_img": "https://example.com/reference.png"
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

- **Few-step distilled model.** The defaults (8 steps / guidance 1.0) are the intended
  operating regime, and match the AI Toolkit training preset for `krea2:o_edit_turbo`.
- At `guidance_scale <= 1.0` the internal guidance is `0`, so the **unconditional pass is
  skipped entirely** and the negative prompt has no effect. Raise `guidance_scale` above
  1.0 to re-enable real CFG (at 2x the cost per step).
- The `ostris/krea2_turbo_training_adapter` is a **training-only** adapter. AI Toolkit
  merges it and then cancels it out at `-1.0` while sampling, so previews come from plain
  Turbo weights. This pipeline never loads it, which is the correct behavior.

- Each reference image is consumed **twice, with different resize policies**:
  1. through the Qwen3-VL text encoder (downscaled to fit ~384x384, bicubic + antialias),
  2. as clean VAE latents appended to the image sequence at `t=0` (area matched to the
     target resolution, snapped to a multiple of 16, bilinear without antialias).
- `kv_cache` is enabled, matching the `krea2:o_edit_turbo` training preset. It is **not** just a
  speed-up: it changes the attention mask so reference tokens attend only to each other.
  A LoRA trained with `kv_cache` on must be inferenced with it on, and vice versa.
- The unconditional (negative) pass receives the **same** reference images. A text-only
  negative would change the CFG direction.
- Up to **3** reference images, matching the AI Toolkit sampling cap.

## Preview-matching notes (training preview vs inference mismatch)

- Start from **8 steps / guidance 1.0** — the AI Toolkit defaults for `krea2:o_edit_turbo`.
- Width/height are floored to a multiple of **16**.
- The model needs roughly **33 GiB** of VRAM in bf16 (a ~12.8B transformer plus a 4B text
  encoder). This pipeline does not quantize, while AI Toolkit training defaults to float8; that
  is a known and deliberate parity delta.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to the divisor).
- Match `sample_steps`, and remember `guidance_scale` is 0-normalized.
- Match `loras[].network_multiplier`.
- Confirm the LoRA was trained for `krea2:o_edit_turbo` (safetensors metadata `ss_base_model_version` is `krea2`).

## Related

- [Krea 2 Turbo](../krea2-turbo/)
- [Krea 2 Edit](../krea2-o-edit/)
