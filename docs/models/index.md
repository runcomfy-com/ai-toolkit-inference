---
title: "Model Catalog"
description: "Supported model IDs in ai-toolkit-inference (Diffusers LoRA inference for ostris/ai-toolkit-trained LoRAs). One page per model with defaults, required inputs, and parity notes."
keywords: "ai-toolkit lora inference models, diffusers inference pipelines, model ids, control image models, video t2v i2v models, preview vs inference mismatch"
permalink: /models/
---

← [Docs Home](../) · [HTTP API](../api/) · [Troubleshooting](../troubleshooting/)

# Model Catalog

This is the index of supported **API `model` ids** for `POST /v1/inference`.

Each model page is meant to be a small **engineering note**: defaults that affect outputs, required inputs (e.g. `ctrl_img`), and the common causes of **AI Toolkit preview vs inference mismatch**.

If you don’t know which model ids your server build supports, call `GET /v1/models` (see [HTTP API](../api/)).

**Run in the cloud (optional):** If you want to reproduce the examples on this page in a pinned runtime without local CUDA/driver setup (and reduce preview‑vs‑inference drift), run it via RunComfy’s Cloud AI Toolkit (Train + Inference).  👉 You can open it here: **[Cloud AI Toolkit (Train + Inference)](https://www.runcomfy.com/trainer/ai-toolkit/app)**

## Naming notes

- URL slugs use hyphens; API `model` ids use underscores.
  - Example: `/models/wan22-14b-i2v/` ↔ `model="wan22_14b_i2v"`
- If you’re integrating this into another app, treat the code registry as canonical: `src/schemas/models.py` + `src/pipelines/__init__.py`.

## Preview-matching knobs (what usually changes outputs)

If you’re trying to reproduce AI Toolkit training **sample previews**, start from the relevant model page and align:

- model id / base checkpoint
- steps + guidance
- width/height after snapping
- control image wiring (for edit/I2V models)
- LoRA loading mode (dynamic adapters vs fused/merged)

## Image generation

| Model page | API model id | Base checkpoint | Notes |
|---|---|---|---|
| [FLUX (FLUX.1-dev)](flux/) | `flux` | `black-forest-labs/FLUX.1-dev` | Pipeline file is named `flux_dev.py`, but the API model id is `flux`. |
| [FLUX.2 (FLUX.2-dev)](flux2/) | `flux2` | `black-forest-labs/FLUX.2-dev` | Uses AI Toolkit components and merges LoRA into the transformer at load time (scale is not dynamically adjustable). |
| [FLUX.2-klein 4B](flux2-klein-4b/) | `flux2_klein_4b` | `black-forest-labs/FLUX.2-klein-base-4B` | Uses AI Toolkit components + Qwen3 text encoder; CFG/negative prompt are active. |
| [FLUX.2-klein 9B](flux2-klein-9b/) | `flux2_klein_9b` | `black-forest-labs/FLUX.2-klein-base-9B` | Uses AI Toolkit components + Qwen3 text encoder; CFG/negative prompt are active. |
| [Flex.1 (alpha)](flex1/) | `flex1` | `ostris/Flex.1-alpha` | Implemented on top of `diffusers.FluxPipeline` (same family as FLUX.1-dev). |
| [Flex.2](flex2/) | `flex2` | `ostris/Flex.2-preview` | Uses `fuse_lora` (weights are merged), so LoRA scale is fixed after load; changing `loras[].network_multiplier` requires reload. |
| [SDXL 1.0](sdxl/) | `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` | Uses a DDPMScheduler config aligned to AI Toolkit defaults. |
| [SD 1.5](sd15/) | `sd15` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Uses a DDPMScheduler config aligned to AI Toolkit defaults. |
| [Qwen Image](qwen-image/) | `qwen_image` | `Qwen/Qwen-Image` | Guidance is passed as `true_cfg_scale` in Diffusers. |
| [Qwen Image (2512)](qwen-image-2512/) | `qwen_image_2512` | `Qwen/Qwen-Image-2512` | Uses `fuse_lora` (weights are merged), so LoRA scale is not dynamically adjustable. |
| [Z-Image Turbo](zimage-turbo/) | `zimage_turbo` | `Tongyi-MAI/Z-Image-Turbo` | Few-step model: the defaults (8 steps / CFG 1.0) are part of the model’s intended regime. |
| [Z-Image De-Turbo](zimage-deturbo/) | `zimage_deturbo` | `Tongyi-MAI/Z-Image-Turbo` | Assembles the pipeline from separate components (transformer from `ostris/Z-Image-De-Turbo`). |
| [Chroma](chroma/) | `chroma` | `lodestones/Chroma1-Base` | Requires AI Toolkit for the custom pipeline and quantization path. LoRA is merged before quantization. |
| [HiDream I1](hidream/) | `hidream` | `HiDream-ai/HiDream-I1-Full` | Heavy pipeline: loads Llama-3.1-8B-Instruct as an extra text encoder and fuses LoRA into the transformer. |
| [Lumina Image 2.0](lumina2/) | `lumina2` | `Alpha-VLLM/Lumina-Image-2.0` | Uses a FlowMatch scheduler config aligned to AI Toolkit’s Lumina2 sampler defaults. |
| [OmniGen2](omnigen2/) | `omnigen2` | `OmniGen2/OmniGen2` | Optional reference images (up to 3 via `ctrl_img(_1..3)`). Applies chat template for preview matching. |

## Editing / control-image

| Model page | API model id | Base checkpoint | Notes |
|---|---|---|---|
| [FLUX Kontext (FLUX.1-Kontext-dev)](flux-kontext/) | `flux_kontext` | `black-forest-labs/FLUX.1-Kontext-dev` | Requires a control image (`ctrl_img`). The server resizes it to match the requested output size. |
| [Qwen Image Edit](qwen-image-edit/) | `qwen_image_edit` | `Qwen/Qwen-Image-Edit` | Requires a control image (`ctrl_img`). Prompt encoding depends on the control image. |
| [Qwen Image Edit Plus (2509)](qwen-image-edit-plus/) | `qwen_image_edit_plus` | `Qwen/Qwen-Image-Edit-2509` | Multi-image edit. Supports up to 3 control images (`ctrl_img_1..3`). Prompt encoding uses those images. |
| [Qwen Image Edit Plus (2511)](qwen-image-edit-plus-2511/) | `qwen_image_edit_plus_2511` | `Qwen/Qwen-Image-Edit-2511` | Multi-image edit with `fuse_lora` (weights are merged). Supports up to 3 control images. |
| [HiDream E1 (HiDream-E1-Full)](hidream-e1/) | `hidream_e1` | `HiDream-ai/HiDream-E1-Full` | Image editing. Requires `ctrl_img`. Uses fused LoRA (reload required for different scales). |

## Video generation

| Model page | API model id | Base checkpoint | Notes |
|---|---|---|---|
| [LTX-2](ltx2/) | `ltx2` | `Lightricks/LTX-2` | Unified T2V/I2V: if you provide `ctrl_img` it runs I2V; otherwise T2V. Outputs frames + audio (MP4). LoRA is converted and fused. |
| [Wan 2.1 T2V (14B)](wan21-14b/) | `wan21_14b` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Text-to-video. Uses `diffusers.WanPipeline`. |
| [Wan 2.1 T2V (1.3B)](wan21-1b/) | `wan21_1b` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Text-to-video. Smaller checkpoint (1.3B) with the same API surface as 14B. |
| [Wan 2.1 I2V (14B)](wan21-i2v-14b/) | `wan21_i2v_14b` | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | Image-to-video. Requires `ctrl_img`. The 480p variant `wan21_i2v_14b480p` uses the same logic but a different base checkpoint. Also supports `wan21_i2v_14b480p`. |
| [Wan 2.2 T2V (A14B)](wan22-14b-t2v/) | `wan22_14b_t2v` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | MoE LoRA format: `loras` with `transformer: "low"` / `"high"` (either side optional). |
| [Wan 2.2 I2V (A14B)](wan22-14b-i2v/) | `wan22_14b_i2v` | `ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16` | Requires AI Toolkit for the custom Wan22Pipeline + first-frame conditioning. Uses MoE LoRA format and requires `ctrl_img`. |
| [Wan 2.2 TI2V (5B)](wan22-5b/) | `wan22_5b` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Tier-2 model: requires AI Toolkit. Supports both T2V and I2V (provide `ctrl_img` for I2V). |
