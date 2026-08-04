---
title: "MiniMax-H3 LoRA Inference (AI Toolkit-trained, video + audio)"
description: "Run minimax_h3 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline. Fixed 24 fps timeline, 17n+5 frame grid, joint stereo audio, and training preview vs inference parity notes."
keywords: "minimax h3 lora inference, MiniMax-H3 pipeline, ai-toolkit minimax_h3 inference, video with audio generation, training preview vs inference mismatch, minimax_h3"
permalink: /models/minimax-h3/
---

← [Docs Home](../../) · [Model Catalog](../) · [HTTP API](../../api/) · [Troubleshooting](../../troubleshooting/)
# MiniMax-H3 LoRA Inference (AI Toolkit-trained, video + audio)

**API model id:** `minimax_h3`
**URL slug:** `minimax-h3`

This page documents the **reference inference pipeline** for `minimax_h3` (MiniMax-H3). It is the first model in this catalog that generates **video and synchronized stereo audio in the same pass**.

Unlike most entries here, this pipeline does **not** wrap a Diffusers pipeline. There is no Diffusers implementation for MiniMax-H3, so the four components are built by hand and handed to **ai-toolkit's own released sampler** — the same code the trainer calls to render preview samples. That is the entire parity story: the dual video/audio sigma schedules, the packed-sequence layout, keyframe conditioning and the decode path cannot drift from training, because they are the same code.

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy's Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/minimax_h3.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/minimax_h3.py) |
| Base checkpoint | `Comfy-Org/MiniMax-H3` (fl2va partition, ~42.5 GB) |
| Defaults | `width=768`, `height=768`, `num_frames=107`, `sample_steps=28`, `guidance_scale=1.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Frame count | Snapped **down** to the `17n + 5` grid: 5, 22, 39, 56, 73, 90, 107, 124 |
| FPS | **Fixed at 24.** Any other value is rejected |
| Control image | Optional — absent = text-to-video, present = first-frame image-to-video |
| Video | Yes, with **joint stereo audio** at 32 kHz |
| Negative prompt | **Not supported** — the model is guidance-distilled and has no unconditional branch |
| LoRA scale behavior | Attached as a **live adapter**, never merged (see below) |
| Needs AI Toolkit | **Required** — the sampler and the quantized-weight loader both come from it |

## Reference implementation (source of truth)

- **Pipeline implementation:** [`src/pipelines/minimax_h3.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/minimax_h3.py)
- **Shared behaviors (snapping, seeding, LoRA base logic):** [`src/pipelines/base.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/base.py)
- **Request schema (parameter names):** [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)
- **Model ids (enum):** [`src/schemas/models.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/models.py)
- **Pipeline registry (model → class):** [`src/pipelines/__init__.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/__init__.py)
- **Sampler (in ai-toolkit):** `extensions_built_in/diffusion_models/minimax_h3/src/pipeline.py`

## Minimal API request

```json
{
  "model": "minimax_h3",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a woman holding a coffee cup, in a beanie, sitting at a cafe",
      "width": 768,
      "height": 768,
      "seed": 42,
      "sample_steps": 28,
      "guidance_scale": 1.0,
      "num_frames": 107,
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

- **The LoRA is attached live, not merged.** Merging into the int8-ConvRot base is dequantize → add → requantize, which resamples every row scale. Training never merges either — its `LoRAModule.forward` is `org_forward(x) + lora_output * scale`. Merging here would inject a requantization round the trainer never performed, and the resulting drift looks exactly like "quantization is lossy" when you eyeball a comparison.
- **The adapter attaches to 258 modules**, not just the attention and MLP projections: `blocks.N.adaln_proj.linear` and the whole `token_refiner` subtree are targets too. A `blocks.*` prefix filter would silently drop 58 of them and still appear to work.
- **Weights are pre-quantized and load as-is**: int8 ConvRot DiT (~21 GB) and an NVFP4 AWQ Qwen3-VL text encoder (~16 GB). The defaults match the shipped files, so nothing is re-quantized on load.
- **Guidance is ignored.** The checkpoint is CFG-distilled and the sampler has no unconditional branch, so `guidance_scale` is accepted and discarded, and a negative prompt has nowhere to go.
- **Single-frame (image) mode is not exposed.** `num_frames=1` makes the sampler return a PIL image rather than the video dict, which the MP4 writer cannot consume.

## Preview-matching notes (training preview vs inference mismatch)

- **Frame counts snap down.** Requesting 123 frames yields 107 — the next grid value below. This is silent apart from a log line, so a "why is my video shorter" report usually traces here.
- **FPS is fixed at 24.** The audio is generated for a 24 fps timeline; muxing at any other rate speeds the video up against its own waveform. The pipeline rejects non-24 rather than producing a desynced file.
- **Resolution floors to a multiple of 32.** 720 becomes 704.
- **Match the training seed convention.** ai-toolkit's sampler uses a CPU generator, and with `walk_seed: true` prompt *i* uses `seed + i`, not `seed`.

## Measured parity

Verified on an H100 against a real production training job (250 steps, rank 16), replaying its exact prompts, seeds and geometry:

| Comparison | PSNR | Audio correlation |
|---|---|---|
| With LoRA vs the job's step-250 sample | **25.3 / 29.5 dB** | 0.982 / 0.989 |
| No LoRA vs the job's step-0 sample | 20.1 / 13.5 dB | 0.995 / 0.884 |
| LoRA at scale 0 vs no LoRA | **byte-identical** | — |

Structure matched exactly in every case: 107 frames, 768×768, 24 fps, stereo 32 kHz, audio length identical to the training sample.

The third row is the useful control: two independent full runs producing byte-identical output means the pipeline is deterministic, so any remaining gap to training is systematic rather than GPU noise.

## What to compare when debugging mismatch

- Confirm the **effective** `num_frames` after snapping — the grid is 17n+5, not arbitrary.
- Confirm the **effective** width/height after the ×32 floor.
- Match `sample_steps`; the sampler runs `num_inference_steps - 1` denoise passes.
- Match `loras[].network_multiplier`. Missing `.alpha` tensors default to **rank**, not 1 — a LoRA saved in PEFT format has them stripped with `alpha == rank`.
- For i2v, match the control image and remember it is resized to the model's canvas before conditioning.
