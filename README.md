<div align="center">

# AI Toolkit Inference (Diffusers LoRA Inference)

Reference **Diffusers LoRA inference pipelines** (plus an optional HTTP server) for LoRAs trained with [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), designed to minimize **AI Toolkit training sample vs inference output drift** so your real inference outputs are consistent with the samples you validated during training, and remain reproducible across environments.

Supports **27+ models** across image generation, editing, and video: FLUX.1/FLUX.2, Flex, SD/SDXL, Qwen Image (and Edit variants), Z-Image, Wan 2.1/2.2, LTX-2, Chroma, HiDream, Lumina2, OmniGen2, and more.

<p>
  <a href="https://ai-toolkit-docs.runcomfy.com/">Docs Home</a> ·
  <a href="https://ai-toolkit-docs.runcomfy.com/models/">Model Catalog</a> ·
  <a href="ComfyUI.md">ComfyUI</a> ·
  <a href="https://www.runcomfy.com/trainer/ai-toolkit/app">Cloud AI Toolkit (Train+Inference)</a> ·
  <a href="#quickstart-http-api">Quickstart</a> ·
  <a href="#api">API</a>
</p>

</div>

## Why this repo exists

You're in the right place if any of these is true:

- You trained a LoRA with **`ostris/ai-toolkit`** and now need a **known-good Diffusers inference pipeline** (Python or HTTP API).
- Your **training samples look good**, but inference in Diffusers / ComfyUI / another stack looks different (same prompt/seed, different output).
- You want a **reference implementation** (defaults + pipeline wiring) instead of guessing "which hidden default changed this time".

Note: if your main blocker is environment drift (CUDA/PyTorch/Diffusers versions, large model downloads, custom pipeline deps), running the same stack in a fixed runtime/container helps. RunComfy provides a managed runtime for AI Toolkit training + inference, but the reference behavior is still defined by the code in this repo.

---

## Docs and model pages

- **Docs Home (GitHub Pages):** https://ai-toolkit-docs.runcomfy.com/
- **Model Catalog (by model id):** https://ai-toolkit-docs.runcomfy.com/models/

Popular model docs (each page includes defaults + what commonly causes preview mismatch for that model):

- [LTX-2 video T2V/I2V](https://ai-toolkit-docs.runcomfy.com/models/ltx2/) (`model="ltx2"`)
- [Wan 2.2 14B T2V MoE LoRA inference](https://ai-toolkit-docs.runcomfy.com/models/wan22-14b-t2v/) (`model="wan22_14b_t2v"`)
- [Wan 2.2 14B I2V MoE LoRA inference (requires ctrl_img)](https://ai-toolkit-docs.runcomfy.com/models/wan22-14b-i2v/) (`model="wan22_14b_i2v"`)
- [Z-Image Turbo LoRA inference (few-step defaults)](https://ai-toolkit-docs.runcomfy.com/models/zimage-turbo/) (`model="zimage_turbo"`)
- [FLUX.2-dev LoRA inference](https://ai-toolkit-docs.runcomfy.com/models/flux2/) (`model="flux2"`)
- [FLUX Kontext LoRA inference (control-image edit)](https://ai-toolkit-docs.runcomfy.com/models/flux-kontext/) (`model="flux_kontext"`)
- [Flex.1 LoRA inference](https://ai-toolkit-docs.runcomfy.com/models/flex1/) (`model="flex1"`)
- [Qwen Image LoRA inference (including Edit variants)](https://ai-toolkit-docs.runcomfy.com/models/qwen-image/) (`model="qwen_image"` and variants)
- [SDXL LoRA inference](https://ai-toolkit-docs.runcomfy.com/models/sdxl/) (`model="sdxl"`)

---

## Versioning / ai-toolkit alignment

This repo publishes tags and releases. Each tag corresponds to a specific **ai-toolkit** version as defined by its `version.py`.  
Since **ai-toolkit** does not publish tags or releases, we pin and document the **exact ai-toolkit commit** that contains that `version.py`.

| ai-toolkit-inference tag | ai-toolkit version (version.py) | ai-toolkit commit |
| --- | --- | --- |
| `v0.7.19.202601281` | `0.7.19` | [`73dedbf662ca604a3035daff2d2ba4635473b7bd`](https://github.com/ostris/ai-toolkit/commit/73dedbf662ca604a3035daff2d2ba4635473b7bd) |

---

## What this repo provides

This repo has two parts that work together:

- **`src/`** — the runnable inference implementation:
  - request/response schema (the parameters you actually pass)
  - model registry + defaults (what changes outputs)
  - async request lifecycle (queue → status → result)
  - per-model pipelines implemented in Diffusers
- **`docs/`** — developer docs:
  - a model catalog (one page per model id / pipeline family)
  - model-specific preview-mismatch notes and recommended starting settings
  - links back to the exact code that runs

If you only read one thing: treat `src/` as the source of truth.

---

## ComfyUI Custom Nodes

This repo can be used as a ComfyUI custom node pack. Install via **ComfyUI-Manager** for automatic dependency setup (including `ostris/ai-toolkit` for extended models).

See: [`ComfyUI.md`](ComfyUI.md)

---

## Quickstart HTTP API

This is the smallest runnable path if you just want an HTTP endpoint for Diffusers LoRA inference.

### 0) Install (once)

See: [Installation](#installation)

### 0.5) Check supported models (optional, but useful)

```bash
curl http://localhost:8000/v1/models
```

### 1) Put your LoRA weights where the server expects them

The API takes `loras[].path` and resolves local paths under `WORKFLOWS_BASE_PATH`:

```
{WORKFLOWS_BASE_PATH}/{loras[].path}
```

Path resolution and validation live in:
- [`src/api/v1/inference.py`](src/api/v1/inference.py)

Notes:
- `loras` is required for all requests.
- `loras[].path` must include the full filename (e.g. `my_lora.safetensors`).
- Non‑MoE models accept **exactly one** LoRA item.
- Wan 2.2 14B (T2V/I2V) uses **MoE format** with `transformer: "low"` / `"high"` (see the API section).
- `loras[].path` can be a URL; the server will download and cache it.

### 2) Run the server

```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Entry point:
- [`src/server.py`](src/server.py)

Settings / environment variables:
- [`src/config.py`](src/config.py)

FastAPI docs (interactive):
- `GET /docs` (Swagger UI)
- `GET /redoc`

### 3) Submit an inference request (example: Z-Image Turbo)

```bash
curl -X POST "http://localhost:8000/v1/inference" \
  -H "Content-Type: application/json" \
  -d '{
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
        "neg": ""
      }
    ]
  }'
```

You'll get a `request_id` plus `status_url` and `result_url`, then poll:

- `GET /v1/requests/{request_id}/status`
- `GET /v1/requests/{request_id}/result`

Request schema (authoritative):
- [`src/schemas/request.py`](src/schemas/request.py)

Response schema:
- [`src/schemas/response.py`](src/schemas/response.py)

### Where the outputs go

Outputs are written under `OUTPUT_BASE_PATH/` as local files **prefixed by** the `request_id`:

- Images: `OUTPUT_BASE_PATH/{request_id}_output_{i}.jpg`
- Videos: `OUTPUT_BASE_PATH/{request_id}_output_{i}.mp4`

The result endpoint returns local `file_path` values for images/videos (no object storage integration by default).

---

## Installation

Requirements:

- Python >= 3.10
- CUDA-capable GPU with sufficient VRAM (CPU can work for some models, but will be slow)
- `ostris/ai-toolkit` (optional; required for extended models—see below)

```bash
# Install PyTorch with CUDA (adjust cu126 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install inference dependencies
pip install -r requirements-inference.txt
```

### ai-toolkit for extended models

Some pipelines (FLUX.2, Chroma, HiDream, OmniGen2, Wan 2.2 I2V/5B, LTX-2) require custom classes from `ostris/ai-toolkit`.

**Automatic setup (recommended):**

Run the install script to clone ai-toolkit into `vendor/ai-toolkit`:

```bash
python install.py
```

The code automatically detects `vendor/ai-toolkit` at runtime—no environment variable needed.

**Manual setup (advanced):**

If you prefer to manage ai-toolkit separately, clone it anywhere and set `AI_TOOLKIT_PATH`:

```bash
git clone https://github.com/ostris/ai-toolkit.git /path/to/ai-toolkit
export AI_TOOLKIT_PATH=/path/to/ai-toolkit
```

Tip: if a model requires ai-toolkit and it's missing, you'll see an `ImportError` referencing `extensions_built_in...` or `toolkit...`.

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `BASE_URL` | Used to build `status_url` / `result_url` in responses | `http://localhost:8000` |
| `DEVICE` | Device: `cuda` or `cpu` | `cuda` |
| `ENABLE_CPU_OFFLOAD` | Enable model CPU offload (helps fit big models) | `false` |
| `WORKFLOWS_BASE_PATH` | Base path for LoRA weight directories | `/app/ai-toolkit/lora_weights` |
| `OUTPUT_BASE_PATH` | Output path for images/videos | `/tmp/inference_output` |
| `AI_TOOLKIT_PATH` | Path to ai-toolkit (only needed for some models) | auto-detected (`vendor/ai-toolkit` if present) |

For the full set of settings (e.g. `DEBUG`, `HF_TOKEN`, `MODEL_CACHE_DIR`, `INFERENCE_TIMEOUT`), see:
- [`src/config.py`](src/config.py)

---

## API

### Endpoints

- `POST /v1/inference` — submit an async inference request
- `GET /v1/requests/{request_id}/status` — returns: `in_queue`, `in_progress`, `succeeded`, `failed`
- `GET /v1/requests/{request_id}/result` — returns the generated images/videos (local file paths)
- `GET /v1/models` — list supported model IDs + defaults

Implementation:
- routes + validation: [`src/api/v1/inference.py`](src/api/v1/inference.py)

### Discover supported model IDs and defaults

If you're unsure which `model` values are accepted (or what defaults a model uses), call:

- `GET /v1/models`

The canonical list in code:
- model enum: [`src/schemas/models.py`](src/schemas/models.py)
- registry mapping: [`src/pipelines/__init__.py`](src/pipelines/__init__.py)

### MoE LoRA format (Wan 2.2 14B models)

For `model="wan22_14b_t2v"` and `model="wan22_14b_i2v"`, `loras` must use MoE format:

```json
{
  "model": "wan22_14b_t2v",
  "loras": [
    { "path": "my_wan_lora/low_noise.safetensors", "transformer": "low", "network_multiplier": 1.0 },
    { "path": "my_wan_lora/high_noise.safetensors", "transformer": "high", "network_multiplier": 1.0 }
  ],
  "prompts": [
    { "prompt": "a cinematic shot", "width": 1280, "height": 720, "seed": 42 }
  ]
}
```

If you send MoE format to a non-MoE model (or multiple LoRAs to a single-LoRA model), the API will return a 400 with details.

---

## Troubleshooting training preview vs inference mismatch

AI Toolkit "Samples" are generated by a specific inference graph: base model variant + scheduler/timestep logic + guidance behavior + prompt encoding + resolution rules + LoRA injection + seed handling.

If your inference environment changes any of those (even with the same prompt/seed), results can drift. This tends to show up most aggressively on:

- **few-step / distilled models** (small graph changes become visible quickly)
- **editing / control-image pipelines** (preprocessing and conditioning wiring matters)
- **models with non-standard guidance implementations**

A pragmatic checklist (common mismatch causes):

- **Resolution snapping**: width/height are floored to a multiple of `resolution_divisor`.  
  See: [`src/pipelines/base.py`](src/pipelines/base.py)
- **Seed semantics**: global seeding + a CPU generator for sampling.  
  See: [`src/pipelines/base.py`](src/pipelines/base.py)
- **LoRA application mode**: adapters vs `fuse_lora` vs model-specific merges.  
  Default behavior lives in: [`src/pipelines/base.py`](src/pipelines/base.py)
- **Control inputs**: some models require `ctrl_img` (or `ctrl_img_1..3`).  
  Validation lives in: [`src/api/v1/inference.py`](src/api/v1/inference.py)

If you're trying to reproduce the preview you validated during training:
1) start with the reference server pipeline for your model, then
2) customize one variable at a time.

For "by model" notes: https://ai-toolkit-docs.runcomfy.com/models/

---

## Source of truth

If you're integrating these pipelines into your own app (instead of running the server as-is), these are the files that define behavior:

- Base behaviors (seed, resolution divisor, LoRA loading):  
  [`src/pipelines/base.py`](src/pipelines/base.py)
- Model registry and mapping (`model` → pipeline class):  
  [`src/pipelines/__init__.py`](src/pipelines/__init__.py)
- API request schema (parameter names you actually pass):  
  [`src/schemas/request.py`](src/schemas/request.py)
- API routes and validation (single LoRA vs MoE, control image requirements):  
  [`src/api/v1/inference.py`](src/api/v1/inference.py)

---

## Development

```bash
# Run tests
pytest

# Run with hot reload
python -m uvicorn src.server:app --reload
```

---

## Docker (note)

The included `Dockerfile` may be tailored to a specific production runtime and may not be a drop-in build for all environments.

If you just want to run the server locally, follow:
- [Quickstart HTTP API](#quickstart-http-api)

If you need a portable container build, use this repo as the source of truth and create a minimal CUDA-enabled image that:
- installs PyTorch + `requirements-inference.txt`
- sets `WORKFLOWS_BASE_PATH` and `OUTPUT_BASE_PATH`
- runs `python -m uvicorn src.server:app ...`

---

## FAQ

### "My AI Toolkit LoRA has no effect in Diffusers"

Most often it's not "the scale is wrong", it's one of:
- wrong base model variant
- LoRA not applied to the expected modules
- you're running a different pipeline family than what the trainer sampled with

A reliable starting point is to run through the server once, then mirror the pipeline code.

### "Training samples look better than inference"

Treat this as an inference-graph mismatch problem. Verify steps/guidance, resolution snapping, LoRA loading mode (adapter vs fuse), and any required control inputs. Then check the model page for model-specific mismatch causes.

### "ComfyUI vs Diffusers mismatch"

Different stacks often implement slightly different step semantics, schedulers, or LoRA application order. This repo is meant to give you a concrete Diffusers reference to compare against.

---

## References

- Ostris AI Toolkit: https://github.com/ostris/ai-toolkit
- Hugging Face Diffusers: https://github.com/huggingface/diffusers
