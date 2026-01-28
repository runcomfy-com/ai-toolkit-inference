---
title: "HTTP API Reference — ai-toolkit-inference"
description: "FastAPI/REST endpoints and request/response formats for ai-toolkit-inference: run LoRAs trained with ostris/ai-toolkit using per-model Diffusers inference pipelines (image, control-image editing, video; ctrl_img; Wan2.2 MoE LoRA)."
keywords: "ai-toolkit inference api, diffusers lora inference api, fastapi lora inference, rest api lora inference, post /v1/inference, ctrl_img, wan2.2 moe"
permalink: /api/
---

← [Docs Home](../) · [Model Catalog](../models/) · [Troubleshooting](../troubleshooting/)


# HTTP API Reference

The **ai-toolkit-inference** server exposes a small **HTTP/REST API** for running **LoRAs trained with `ostris/ai-toolkit`** on top of curated **Diffusers pipelines**.

For per-model behavior and defaults, cross-check the [Model Catalog](../models/).

## Quick navigation

- [Endpoints](#endpoints)
- [Async lifecycle](#async-lifecycle)
- [Request schema](#request-schema)
- [Control image formats](#control-image-formats)
- [Responses](#responses)
- [Error codes](#error-codes-common)
- [Practical examples](#practical-examples)

This API is intentionally simple:

- One endpoint to start an inference request
- Two endpoints to poll status + fetch results
- One endpoint to list supported models and their defaults

Note (optional): if your main blocker is **environment drift** (CUDA/PyTorch/Diffusers versions, large model downloads, custom pipeline deps), it helps to run the server in a fixed runtime/container. RunComfy provides a managed runtime for AI Toolkit training + inference, but any reproducible GPU environment works — the reference behavior is still defined by this repo.

---

## Base URL

The server uses `BASE_URL` (see `src/config.py`) to construct `status_url` and `result_url`. Default is usually:

- `http://localhost:8000`

---

## Endpoints

### Health

- `GET /health` → `{ "status": "healthy" }`

### Start an inference

- `POST /v1/inference`

This creates an async task and returns URLs you can poll.

### Poll status

- `GET /v1/requests/{request_id}/status`

### Fetch result

- `GET /v1/requests/{request_id}/result`

### List supported models + defaults

- `GET /v1/models`

OpenAPI / interactive docs (served by FastAPI):

- `GET /docs`
- `GET /openapi.json`

This is the canonical way to discover:

- valid `model` IDs
- per-model defaults (steps, guidance, resolution divisor, etc.)
- whether a model requires a control image
- whether a model requires ai-toolkit for custom pipelines

---

## Async lifecycle

1) `POST /v1/inference` → returns `request_id`, `status_url`, `result_url`
2) Poll `GET /v1/requests/{request_id}/status`
3) Fetch `GET /v1/requests/{request_id}/result`

Notes:

- The implementation uses an **in-memory** task store (restart wipes state).
- The executor runs one inference at a time (global lock) to reduce VRAM contention.

---

## Request schema

Source of truth: [`src/schemas/request.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/schemas/request.py)

### Top-level fields

```json
{
  "model": "flux",
  "hf_token": "hf_*** (optional)",
  "trigger_word": "sks (optional)",
  "prompts": [
    {
      "prompt": "[trigger] a cinematic photo",
      "neg": "",
      "width": 1024,
      "height": 1024,
      "sample_steps": 25,
      "guidance_scale": 4.0,
      "seed": 42
    }
  ],
  "loras": [
    {
      "path": "my_job/my_job.safetensors",
      "network_multiplier": 1.0
    }
  ]
}
```

Field details:

- `model` (required): one of the supported **API model ids**.
- `loras` (required): list of LoRA items.
  - Non‑MoE models require **exactly one** LoRA item.
  - Wan 2.2 14B models require **MoE format** (see below).
  - Each item supports:
    - `path` (required): local file path (relative to `WORKFLOWS_BASE_PATH`) or a URL.
    - `transformer` (optional): `low` or `high` (MoE models only).
    - `network_multiplier` (optional): LoRA scale (defaults to model config).
- `hf_token` (optional): Hugging Face token for gated models.
- `trigger_word` (optional): if provided, the server replaces `[trigger]` in every prompt.
- `prompts` (required): list of prompt items.

### LoRA path resolution

The server resolves each LoRA path as:

`{WORKFLOWS_BASE_PATH}/{loras[].path}` (unless the path is a URL or absolute)

Implementation: `src/api/v1/inference.py`.

Common layouts:

- `workflows/my_job/my_job.safetensors` → `loras: [{"path": "my_job/my_job.safetensors"}]`
- `workflows/my_lora.safetensors` → `loras: [{"path": "my_lora.safetensors"}]`

### MoE LoRA format (Wan 2.2 14B)

For:

- `model="wan22_14b_t2v"`
- `model="wan22_14b_i2v"`

`loras` must include **transformer-tagged** items for `low`/`high`:

```json
"loras": [
  {"path": "low_noise.safetensors", "transformer": "low", "network_multiplier": 1.0},
  {"path": "high_noise.safetensors", "transformer": "high", "network_multiplier": 1.0}
]
```

You may provide only one side if you trained only one LoRA.

---

## Prompt item fields

Prompt items are objects inside `prompts: [...]`.

Common fields:

- `prompt` (required): the positive prompt.
- `trigger_word` (optional): per-prompt override for `[trigger]` replacement.
- `neg` (optional): negative prompt.
- `width`, `height` (optional): requested output size.
  - The server **floors** both to a multiple of the model’s `resolution_divisor`.
- `sample_steps` (optional): steps.
- `guidance_scale` (optional): CFG scale.
- `sampler` (optional): sampler name (pipeline-specific; most pipelines ignore this for now).
- `seed` (optional): integer seed.
  - If omitted, the server uses the pipeline default seed (usually `42`) **plus the prompt index** (0-based).
  - To force randomization, set `seed: -1` (the server generates a random seed and returns it in outputs).

Video-only fields (only respected by video pipelines):

- `num_frames`
- `fps`

Control image fields:

- `ctrl_img`: a single control image
- `ctrl_img_1`, `ctrl_img_2`, `ctrl_img_3`: additional control images (used by multi-image edit models)

---

## Control image formats

Source of truth for loading: [`src/libs/image_utils.py`](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/libs/image_utils.py)

The server accepts a **string**. If it starts with `http`, it’s treated as a URL; otherwise it’s treated as base64.

1) URL

```json
"ctrl_img": "https://example.com/image.png"
```

2) Base64 (raw or data URL)

```json
"ctrl_img": "iVBORw0KGgoAAAANSUhEUgAA..."
```

```json
"ctrl_img": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
```

---

## Responses

Source of truth: `src/schemas/response.py`.

### `POST /v1/inference` response

```json
{
  "request_id": "b3e6e2e0-...",
  "status": "queued",
  "status_url": "http://localhost:8000/v1/requests/b3e6e2e0-.../status",
  "result_url": "http://localhost:8000/v1/requests/b3e6e2e0-.../result",
  "created_at": "2026-01-17T00:00:00Z"
}
```

### Status response

```json
{
  "request_id": "b3e6e2e0-...",
  "status": "in_queue|in_progress|succeeded|failed|canceled",
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "error": null
}
```

### Result response

```json
{
  "request_id": "b3e6e2e0-...",
  "status": "succeeded",
  "outputs": {
    "images": [
      {"format": "jpeg", "width": 1024, "height": 1024, "file_path": "/tmp/inference_output/b3e6e2e0-..._output_0.jpg", "seed": 42}
    ],
    "videos": [
      {"format": "mp4", "width": 832, "height": 480, "num_frames": 41, "fps": 16, "file_path": "/tmp/inference_output/b3e6e2e0-..._output_0.mp4", "seed": 42}
    ]
  },
  "error": null,
  "created_at": "...",
  "finished_at": "...",
  "metadata": {"timings": {"load_pipeline": 12.3, "prompts": [...]}}
}
```

Notes:

- If you query `.../result` before the task finishes, `outputs` may be empty/null and `status` will indicate the current state.
- `file_path` is a **local path on the server**.
- By default there is no static file hosting / object storage integration.

---

## Error codes (common)

Errors are returned with structured JSON. Common `code` values include:

- `MODEL_NOT_FOUND`
- `LORA_FILE_REQUIRED`
- `CONTROL_IMAGE_REQUIRED`
- `MOE_FORMAT_REQUIRED`
- `SINGLE_LORA_ONLY`
- `SINGLE_MOE_CONFIG_ONLY`
- `LORA_NOT_FOUND`
- `REQUEST_NOT_FOUND`
- `MOE_FORMAT_NOT_SUPPORTED`

Implementation: `src/api/v1/inference.py`.

---

## Practical examples

### 1) Text-to-image (FLUX)

```bash
curl -X POST http://localhost:8000/v1/inference \
  -H 'content-type: application/json' \
  -d '{
    "model": "flux",
    "trigger_word": "sks",
    "loras": [{"path": "my_flux_job/my_flux_job.safetensors", "network_multiplier": 1.0}],
    "prompts": [{
      "prompt": "[trigger] a cinematic portrait photo",
      "width": 1024,
      "height": 1024,
      "sample_steps": 25,
      "guidance_scale": 4.0,
      "seed": 42
    }]
  }'
```

### 2) Control-image editing (FLUX Kontext)

```bash
curl -X POST http://localhost:8000/v1/inference \
  -H 'content-type: application/json' \
  -d '{
    "model": "flux_kontext",
    "loras": [{"path": "my_kontext_job/my_kontext_job.safetensors", "network_multiplier": 1.0}],
    "prompts": [{
      "prompt": "make it look like a watercolor",
      "width": 1024,
      "height": 1024,
      "ctrl_img": "https://example.com/input.png",
      "seed": 42
    }]
  }'
```

### 3) Wan 2.2 MoE video (T2V)

```bash
curl -X POST http://localhost:8000/v1/inference \
  -H 'content-type: application/json' \
  -d '{
    "model": "wan22_14b_t2v",
    "loras": [
      {"path": "my_wan_job/low_noise.safetensors", "transformer": "low", "network_multiplier": 1.0},
      {"path": "my_wan_job/high_noise.safetensors", "transformer": "high", "network_multiplier": 1.0}
    ],
    "prompts": [{
      "prompt": "a cinematic shot of a city at night",
      "width": 1280,
      "height": 720,
      "num_frames": 41,
      "fps": 16,
      "sample_steps": 25,
      "guidance_scale": 4.0,
      "seed": 42
    }]
  }'
```

### 4) Python (requests) + polling

```python
import time

import requests


BASE_URL = "http://localhost:8000"

payload = {
    "model": "zimage_turbo",
    "trigger_word": "sks",
    "loras": [{"path": "my_lora_job/my_lora_job.safetensors", "network_multiplier": 1.0}],
    "prompts": [
        {
            "prompt": "[trigger] a photo of a person",
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "sample_steps": 8,
            "guidance_scale": 1.0,
            "neg": "",
        }
    ],
}

# 1) Start request
r = requests.post(f"{BASE_URL}/v1/inference", json=payload, timeout=30)
r.raise_for_status()
job = r.json()
request_id = job["request_id"]

# 2) Poll status
while True:
    s = requests.get(f"{BASE_URL}/v1/requests/{request_id}/status", timeout=30)
    s.raise_for_status()
    status = s.json()["status"]
    if status in ("succeeded", "failed", "canceled"):
        break
    time.sleep(1)

# 3) Fetch result
result = requests.get(f"{BASE_URL}/v1/requests/{request_id}/result", timeout=30)
result.raise_for_status()
print(result.json())
```

Reminder: `file_path` values in results are **local paths on the server** (this repo does not ship object storage/static hosting by default).

---

## Next: model-specific docs

Once you can hit the API successfully, the fastest way to get “preview matching” is to follow the per-model page:

- [Model Catalog](../models/)
- [Troubleshooting](../troubleshooting/)
