# ComfyUI Custom Nodes

This repo can be used directly as a **ComfyUI custom node pack**. When placed in `ComfyUI/custom_nodes/ai-toolkit-inference`, it registers nodes that wrap the pipelines in `src/pipelines/`.

## Installation

### Option 1: ComfyUI-Manager (recommended)

Install via **ComfyUI-Manager** → "Install Custom Nodes" → search for `ai-toolkit-inference`.

ComfyUI-Manager will automatically:
1. Clone this repo into `ComfyUI/custom_nodes/ai-toolkit-inference`
2. Run `install.py`, which:
   - Installs Python dependencies (`requirements-inference.txt`)
   - Clones `ostris/ai-toolkit` into `vendor/ai-toolkit` (required for extended models)

After installation, restart ComfyUI. All nodes—including extended models like FLUX.2, Chroma, HiDream, OmniGen2, LTX-2, and Wan 2.2—should work out of the box.

### Option 2: Symlink (for development)

```bash
ln -s /path/to/ai-toolkit-inference /path/to/ComfyUI/custom_nodes/ai-toolkit-inference
```

### Option 3: Copy

```bash
cp -r /path/to/ai-toolkit-inference /path/to/ComfyUI/custom_nodes/ai-toolkit-inference
```

### Manual dependency installation (for Options 2 & 3)

If you installed manually (symlink or copy), run the install script or install dependencies yourself:

```bash
# Option A: Run the install script (recommended)
cd /path/to/ComfyUI/custom_nodes/ai-toolkit-inference
python install.py

# Option B: Install manually
source /path/to/comfyui-venv/bin/activate
pip install -r /path/to/ai-toolkit-inference/requirements-inference.txt
# For extended models, also clone ai-toolkit:
git clone --depth 1 https://github.com/ostris/ai-toolkit.git /path/to/ai-toolkit-inference/vendor/ai-toolkit
```

Restart ComfyUI. You should see `ai-toolkit-inference` in the custom node import log.

## Verifying the node pack loaded

- In logs, look for the custom node import timing list.
- Via API (if ComfyUI is running):
  - `GET /system_stats` (server + CUDA visibility)
  - `GET /object_info` (all node class types)

## Nodes

All nodes are under the `RunComfy-Inference` category.

### Common inputs (most nodes)

- `prompt` (STRING)
- `negative_prompt` (STRING, optional on most models)
- `width`, `height` (INT)
- `sample_steps` (INT)
- `guidance_scale` (FLOAT)
- `seed` (INT)
- `lora_path` (STRING, optional)
  - One LoRA only at the moment (no stacking / no comma-separated lists).
  - Can be a local path or a URL. URLs are downloaded to a local cache (`AITK_LORA_CACHE_DIR`, default `/tmp/lora_cache`).
  - Hugging Face tip: if you paste a `.../blob/...` URL, we automatically rewrite it to `.../resolve/...` so the file downloads correctly.
- `lora_scale` (FLOAT)
- `hf_token` (STRING, optional)

### Control-image models

Some nodes require an `IMAGE` input as a control/reference image:

- `RCFluxKontext`
- `RCQwenImageEdit`
- `RCQwenImageEditPlus` (+ optional `control_image_2`, `control_image_3`)
- `RCQwenImageEditPlus2511` (+ optional `control_image_2`, `control_image_3`)
- `RCHiDreamE1`
- `RCWan21I2V14B`
- `RCWan21I2V14B480P`
- `RCWan22I2V14B`

`RCOmniGen2` supports reference images optionally (up to 5 control image slots).

### Video models

Video pipelines return an `IMAGE` batch where the batch dimension is frames:

- `RCLTX2`
- `RCWan21T2V14B`, `RCWan21T2V1B`
- `RCWan21I2V14B`, `RCWan21I2V14B480P`
- `RCWan22T2V14B`, `RCWan22I2V14B`
- `RCWan22TI2V5B`

They expose `num_frames` and `fps` as optional inputs.

### Wan 2.2 14B MoE LoRA inputs

`RCWan22T2V14B` and `RCWan22I2V14B` use MoE dual-LoRA inputs:

- `lora_path_high`
- `lora_path_low`

(Instead of using the single `lora_path` string.)

## Available nodes (class types)

- Z-Image:
  - `RCZimageTurbo`
  - `RCZimageDeturbo`
- FLUX:
  - `RCFluxDev`
  - `RCFluxKontext`
  - `RCFlux2`
- Flex:
  - `RCFlex1`
  - `RCFlex2`
- Stable Diffusion:
  - `RCSD15`
  - `RCSDXL`
- Qwen:
  - `RCQwenImage`
  - `RCQwenImage2512`
  - `RCQwenImageEdit`
  - `RCQwenImageEditPlus`
  - `RCQwenImageEditPlus2511`
- Other image models:
  - `RCChroma`
  - `RCHiDream`
  - `RCHiDreamE1`
  - `RCLumina2`
  - `RCOmniGen2`
- Video:
  - `RCLTX2`
  - `RCWan21T2V14B`
  - `RCWan21T2V1B`
  - `RCWan21I2V14B`
  - `RCWan21I2V14B480P`
  - `RCWan22T2V14B`
  - `RCWan22I2V14B`
  - `RCWan22TI2V5B`

## Example workflows

Minimal example workflows are provided in `example_workflows/`:

- `example_workflows/rc_zimage_turbo_minimal.json`
- `example_workflows/rc_zimage_deturbo_minimal.json`
- `example_workflows/rc_sd15_minimal.json`
- `example_workflows/rc_sdxl_minimal.json`
- ...and one `rc_<model>_minimal.json` for each node.

Workflows that require a control image use a `LoadImage` node referencing `aitk_control.png`.
Place that file in `ComfyUI/input/aitk_control.png` (or change the workflow to match your filename).

## Where the code lives

- Node registration: `comfyui_nodes/__init__.py`
- Node implementations:
  - `comfyui_nodes/rc_models.py` (all catalog nodes)
  - `comfyui_nodes/rc_common.py` (shared helpers)

## Notes / common issues

- **Extended models** (FLUX.2, Chroma, HiDream, OmniGen2, LTX-2, Wan 2.2) require `ostris/ai-toolkit`.
  - If you installed via **ComfyUI-Manager**, ai-toolkit is automatically cloned to `vendor/ai-toolkit`.
  - If you see `ImportError: ... from extensions_built_in...` or `from toolkit...`, ai-toolkit is missing. Run `python install.py` from the node pack folder to fix.
  - Advanced users can override the path by setting the `AI_TOOLKIT_PATH` environment variable.
- Large models can require significant disk cache; if downloads fail with "no space left on device", free space in the HuggingFace cache and retry.
