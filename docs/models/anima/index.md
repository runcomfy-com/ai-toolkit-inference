---
title: "Anima LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run anima LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Defaults, required inputs, and training preview vs inference mismatch notes."
keywords: "anima lora inference, Anima diffusers pipeline, ai-toolkit anima inference, training preview vs inference mismatch, anima"
permalink: /models/anima/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# Anima LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `anima`  
**URL slug:** `anima`  
**AI Toolkit training arch:** `anima`

This page documents the **reference inference pipeline** for `anima` — a Cosmos-DiT
flow-matching text-to-image model. It is designed for running **LoRAs trained with**
`ostris/ai-toolkit` while minimizing **training preview vs inference mismatch**.
If you are trying to reproduce AI Toolkit sample previews, treat the code linked below as the source of truth.

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview-vs-inference drift), run it via RunComfy's Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/anima.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/anima.py) |
| Base checkpoint | `circlestone-labs/Anima-Base-v1.0-Diffusers` (self-contained Diffusers *modular* repo — every component is a subfolder) |
| Denoiser | `CosmosTransformer3DModel` (Diffusers) |
| Text encoder | `Qwen3Model` + a learned `AnimaTextConditioner` that fuses the Qwen states with T5 token ids |
| VAE | `AutoencoderKLQwenImage` (f8, 16 latent channels) |
| Defaults | `sample_steps=30`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** (the trainer's bucket divisibility) |
| Control image | No |
| LoRA scale behavior | **Fused into the transformer at load.** Changing `loras[].network_multiplier` triggers a full model reload. |
| Needs AI Toolkit | **Yes** — the sampler uses AI Toolkit's `CustomFlowMatchEulerDiscreteScheduler`. |
| Needs Diffusers ≥ `c943837` | **Yes** — the Anima modular pipeline (`AnimaAutoBlocks`) was added in huggingface/diffusers#13732. Older Diffusers has no Anima module. |
| Gated weights | No — the repo is public (licensed; see the repo's `LICENSE.md`). |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/anima.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/anima.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Upstream model:** `extensions_built_in/diffusion_models/anima/` in [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) + the Anima modular pipeline in [huggingface/diffusers](https://github.com/huggingface/diffusers).

## Minimal API request

```json
{
  "model": "anima",
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

- **`guidance_scale` is RAW — do NOT 0-normalize.** The trainer sets
  `pipeline.guider.guidance_scale = gen_config.guidance_scale` with no `-1`
  (unlike Krea 2, which subtracts 1). A `guidance_scale` here means exactly what
  `sample.guidance_scale` means in a training config. Subtracting 1 would
  under-guide by a full point.
- **Text encoding mirrors the trainer, not the stock Diffusers path.** The pipeline
  builds the *embeds* path (`AnimaCoreDenoiseStep` + `AnimaDecodeStep`) and encodes
  the prompt itself, applying AI Toolkit's **empty-prompt mask fix**: an empty
  prompt (e.g. the unconditional branch under CFG) gets one live attention-mask
  token instead of an all-zero sequence. Diffusers' own `AnimaTextEncoderStep`
  does not do this, so routing the uncond branch through it would drift the CFG
  direction from the training preview.
- **Default negative prompt.** When the request sends an empty `neg`, the pipeline
  falls back to the Anima preset's default
  (`worst quality, low quality, score_1, …`), matching AI Toolkit's UI default.
- LoRA is fused into the transformer weights via the `AnimaLoraLoaderMixin`, so
  `network_multiplier` is fixed after load. A different scale forces a full reload.
  On-disk keys are AI Toolkit's comfy format (`diffusion_model.*`); the mixin's
  `lora_state_dict` converts them and routes to the transformer (and the text
  conditioner, if the adapter trained it).

## Preview-matching notes (training preview vs inference mismatch)

- Start from **30 steps / guidance 4.0** — the AI Toolkit defaults for `anima`.
- Width/height are floored to a multiple of **32**.
- This pipeline runs bf16 without quantization, while AI Toolkit training can use
  float8; that is a known and deliberate parity delta.

## What to compare when debugging mismatch

- Confirm the **effective** width/height after snapping (the server floors to 32).
- Match `sample_steps`, and remember `guidance_scale` is **raw** (no `-1`).
- Match `loras[].network_multiplier`.
- Confirm the LoRA was trained for `anima` (safetensors metadata
  `ss_base_model_version` is `anima`).
