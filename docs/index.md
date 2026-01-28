---
title: "ai-toolkit-inference Docs — AI Toolkit LoRA Inference with Diffusers"
description: "Developer docs for ai-toolkit-inference: reference Diffusers pipelines + async FastAPI server for running LoRAs trained with ostris/ai-toolkit (image, control-image editing, video; ctrl_img; Wan2.2 MoE)."
keywords: "ai-toolkit inference, diffusers lora inference, fastapi lora inference server, preview vs inference mismatch, ctrl_img, wan2.2 moe lora, model catalog"
permalink: /
---

# ai-toolkit-inference Docs

`ai-toolkit-inference` is a set of **reference Diffusers pipelines** plus an **async FastAPI server** for running **LoRAs trained with** `ostris/ai-toolkit`.

The core goal is **training preview ↔ inference parity**. AI Toolkit samples are produced by a specific inference graph (scheduler wiring, resolution rules, LoRA injection, seeding). Small “default” differences between stacks can show up as visibly different outputs.

## Start here

- **Model Catalog:** [Browse supported models](models/)
- **API reference:** [Request format + examples](api/)
- **Troubleshooting:** [Training preview vs inference mismatch](troubleshooting/)
- **Repository README:** [`README.md`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/README.md)

## What you’ll find in these docs

- **One page per API `model` id** (with defaults, required inputs like `ctrl_img`, and preview-matching notes).
- The **HTTP API contract**: request/response schema, async lifecycle, and real curl/Python examples.
- A pragmatic mismatch checklist for common cases: **LoRA not applying**, **control-image wiring**, **resolution snapping**, **Wan 2.2 MoE LoRA format**, etc.

If you’re calling the server and you don’t know which model ids are supported, start with `GET /v1/models` (documented under [HTTP API](api/)).

## What’s in this repo

- `src/pipelines/` — one pipeline implementation per API **model id**
- `src/api/` + `src/server.py` — FastAPI server (`POST /v1/inference`, plus status/result polling)
- `docs/` — this documentation (published as GitHub Pages)

## Quick API shape

1. `POST /v1/inference` → returns `request_id`, `status_url`, `result_url`
2. Poll `GET /v1/requests/{request_id}/status`
3. Fetch `GET /v1/requests/{request_id}/result`

The request schema is defined in [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py).

## Preview vs inference mismatch (the 80/20)

If AI Toolkit training **samples** look good but your inference looks different, the most common causes are:

- **wrong base checkpoint / model id**
- **different steps + guidance regime**
- **different width/height after snapping**
- **different control image preprocessing / wiring**
- **different LoRA loading mode (dynamic adapters vs fused/merged)**

Each model page calls out the model-specific version of these pitfalls.

## Glossary (short)

- **`model` / model id**: the API selector (e.g. `flux2`, `wan22_14b_i2v`). See [Model Catalog](models/) or `GET /v1/models`.
- **`ctrl_img`**: control image input for edit/I2V models. In this server it’s a **string** (URL or base64). See [HTTP API](api/).
- **resolution snapping**: the server floors `width/height` to a multiple of the model’s `resolution_divisor`.
- **`network_multiplier`** (on `loras[]`): LoRA strength. Some pipelines apply it dynamically; some fuse/merge weights and require a reload.

Note: if your main blocker is **environment drift** (CUDA/PyTorch/Diffusers versions, large model downloads, model-specific pipeline deps), it helps to run training + inference in a fixed runtime/container. RunComfy provides a managed runtime for [AI Toolkit](https://www.runcomfy.com/trainer/ai-toolkit/app), but any reproducible GPU environment works — the reference behavior is still defined by this repo.
