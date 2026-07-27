---
name: add-inference-model
description: >
  End-to-end runbook for adding a new model architecture to
  runcomfy-com/ai-toolkit-inference: pipeline implementation, registry
  wiring, ComfyUI nodes, tests, docs, GPU parity verification against real
  production LoRAs, release image build, and the runcomfy-website /
  trainer-contents wiring that actually exposes the model to users. Written
  from the Krea 2 rollout (PR #26, v0.11.0.202607241). Trigger with
  "add <model> to ai-toolkit-inference", "接入 <model> 推理", or when a new
  ai-toolkit ModelArch needs a Diffusers inference path.
---

# Adding a model to ai-toolkit-inference

This repo is the **reference Diffusers inference implementation** for LoRAs
trained by `ostris/ai-toolkit`. Its whole reason to exist is
**training-sample vs inference-output parity** — the images a user saw during
training must reproduce at inference. Every decision below serves that.

Work in phases. Each phase has a gate you must pass before the next. Do not
skip the verification gate — the parity check is the product.

---

## Phase 0 — Understand the model before writing code

Before touching this repo, read the arch in the **training** repo
(`ostris/ai-toolkit`, our fork `InceptionsAI/ai-toolkit`, branch `rc-main`):

```
extensions_built_in/diffusion_models/<arch>/
```

Answer these, quoting file:line — the pipeline is wrong if any is guessed:

1. **Components**: transformer class + params, text encoder (which HF repo,
   which hidden layers), VAE (scale factor, latent channels). → sets
   `resolution_divisor` and the download list.
2. **Sampling contract**: scheduler / sigma schedule, flow-matching time
   convention, how the seed maps to initial noise (device, dtype). → the
   inference loop must match this exactly or output drifts.
3. **CFG convention**: is `guidance_scale` used raw, or normalized? (Krea 2
   does `guidance = max(0, guidance_scale - 1)` — getting this wrong is a
   flat over/under-guidance.) Is the model distilled (cfg=1, single forward)?
4. **Turbo / distilled variants**: separate weights, or base + an adapter?
   If an adapter — is it needed at **inference**? (Krea 2 turbo's
   `assistant_lora` is merged then cancelled at -1.0 during sampling → it is
   TRAINING-ONLY; inference must NOT load it.)
5. **Edit / reference-image mode**: how are control images preprocessed and
   injected? How many? Any per-arch `model_kwargs` (e.g. Krea 2's
   `kv_cache`, `match_target_res`)?
6. **LoRA key format on disk**: run the trace — what do the saved
   `.safetensors` keys literally look like? This decides the merge code.
7. **Is there a native diffusers pipeline?** Check `huggingface/diffusers`
   AND our pinned diffusers commit (`requirements_base.txt` in ai-toolkit —
   the image installs THAT, not this repo's `requirements-inference.txt`,
   see Phase 4). If diffusers has it but our pin predates it, you still can't
   use it without a pin bump — and a pin bump can break other models
   (v0.10.25 broke Qwen edit). Krea 2 chose the ai-toolkit-backed path for
   this reason, plus diffusers had no edit pipeline.

Cross-check the training defaults in the UI:
`ai-toolkit/ui/src/app/jobs/new/options.ts` (per-arch defaults) and
`ai-toolkit/ui/src/helpers/defaultSamples.ts` (global sample defaults:
steps, guidance, walk_seed). These become your `PipelineConfig` values.

> For a heavy dossier, `reference/how-krea2-was-analyzed.md` shows the
> parallel-research approach used for Krea 2.

---

## Phase 1 — Implement the pipeline

### The two hard rules (both are FastAPI-startup landmines)

1. **No heavy imports at module scope in `src/pipelines/<model>.py`.**
   `src/api/v1/inference.py` imports `PIPELINE_REGISTRY` at import time, and
   `src/pipelines/__init__.py:_build_pipeline_registry()` eagerly imports
   every pipeline class **with no try/except**. A top-level
   `import torch` / `import diffusers` / `from extensions_built_in...` in your
   file kills startup for **all** models. Put every heavy import inside a
   method. Existing ai-toolkit-backed pipelines (chroma, flux2, hidream,
   omnigen2, wan22) all do this.

2. **Reaching ai-toolkit code**: import the *leaf* modules by file path, do
   NOT `import extensions_built_in.diffusion_models.<arch>` — that package's
   `__init__` eagerly imports ~21 architectures and drags in
   av/torchaudio/optimum.quanto/toolkit.*. See `_load_aitk_krea2_modules()` /
   `_find_krea2_src_dir()` in `src/pipelines/krea2.py` for the pattern
   (resolves via `settings.ai_toolkit_path`, then `find_spec` on sys.path).
   The production image exposes ai-toolkit via PYTHONPATH, not
   `AI_TOOLKIT_PATH` — `_find_krea2_src_dir` handles both.

### PipelineConfig

`src/pipelines/base.py:PipelineConfig` is the ONLY per-model config surface
`executor._get_pipeline_defaults()` reads. Set: `base_model`,
`resolution_divisor` (= vae_scale_factor × patch_size), `default_steps`,
`default_guidance_scale`, `requires_control_image`, `lora_merge_method`,
`supports_negative_prompt`, `default_width/height`. Values come from Phase 0.

### LoRA merge

Most ai-toolkit-backed models have no diffusers `load_lora_weights`, so use
`LoraMergeMethod.CUSTOM` and merge manually (see `chroma.py`, `flux2.py`).
Traps proven in the Krea 2 rollout:

- **No prefix whitelist.** ai-toolkit's module gate is a *substring* test for
  `"blocks"`. Krea 2's text tower is `txtfusion.layerwise_blocks` /
  `refiner_blocks` — a `startswith("blocks.")` filter silently drops 32 of
  256 modules. Merge key-driven against `transformer.state_dict()`.
- **Missing `.alpha` → default to rank.** ai-toolkit's peft_format writes no
  `.alpha` (alpha == rank), so a missing alpha means internal scale 1.0.
- **Raise on 0 merged.** A wrong-arch LoRA must fail loudly, not silently run
  the base model.
- Handle LoKR too if the arch allows it (`flux2.py:_merge_lokr_to_transformer`).

### Per-request options that can't be read from the LoRA file

If a setting must match how the LoRA was **trained** and is NOT in the LoRA
metadata (ai-toolkit stores only `training_info`, `ss_base_model_version`,
`ss_output_name`, trigger word — never `model_kwargs`), it must be a request
parameter, not a hardcoded default. Krea 2 edit's `kv_cache` /
`match_target_res` flipped on 2026-07-16; adapters trained before need them
off, and `kv_cache` changes the attention mask so a mismatch is silently
wrong. Mechanism: `BasePipeline.apply_model_options(**opts)` (no-op default) +
fields on `InferenceInput` + executor calls it after `get_pipeline`. **It must
also be wired into BOTH ComfyUI entry points** (`_RCAitkBase` and
`RCAITKGenerate`) — the HTTP path and the node path are independent; fixing
one is not fixing the other (Codex caught this on PR #26).

### Own the sampling loop if the vendor pipeline lacks a step callback

ai-toolkit's preview pipelines often have no `callback_on_step_end`, so
ComfyUI progress/cancel would silently do nothing. Port the Euler loop
(≈35 lines) but **import** the subtle math (`predict_velocity`, the mu
schedule, packing, ref RoPE placement) from ai-toolkit — never reimplement it,
or you reintroduce drift.

---

## Phase 2 — Register + surface (the 24-file checklist)

Union of touched files from the Krea 2 merge. `[fatal]` = the model 400s /
startup breaks if skipped.

**Core (required):**
- `[fatal]` `src/schemas/models.py` — `ModelType` enum member(s). Id mirrors
  the training arch with `:` → `_` (e.g. `krea2:o_edit` → `krea2_o_edit`).
  `scripts/request_samples_from_config.py` relies on this exact mapping.
- `[fatal]` `src/pipelines/<model>.py` — the class(es).
- `[fatal]` `src/pipelines/__init__.py` — **4 edits**: `TYPE_CHECKING` block,
  `_LAZY_IMPORTS`, `_MODEL_TYPE_TO_CLASS`, `__all__`.
- `src/services/download_config.py` — allow/ignore patterns + extras (text
  encoder, VAE). **All entries sharing a repo MUST use identical
  allow_patterns** — `_merge_download_tasks` unions them and a `None` on
  either side collapses to a full-repo pull.

**Conditional:**
- `src/schemas/request.py` — only for genuinely new per-request params
  (e.g. `kv_cache`); add a `get_model_options()`.
- `src/pipelines/base.py` — only if adding a shared hook (e.g.
  `apply_model_options`).
- `src/tasks/executor.py` — only to call a new hook.
- `src/api/v1/inference.py:MOE_LORA_MODELS` — only for dual-transformer MoE.

**ComfyUI (3 files, keep in sync):**
- `comfyui_nodes/rc_models.py` — one `_RCAitkBase` subclass per model.
  `RESOLUTION_STEP` MUST equal `resolution_divisor`. Declare `MODEL_OPTIONS`
  for any per-request options.
- `comfyui_nodes/__init__.py` — 3 edits: import, `NODE_CLASS_MAPPINGS`,
  `NODE_DISPLAY_NAME_MAPPINGS`.
- `comfyui_nodes/rc_latent_workflow.py` — **2 edits that MUST match**:
  `ALL_PIPELINES` and `ctor_map`. A model in the first but not the second
  raises `ValueError: Unknown pipeline` at execution.

**Docs / examples (non-fatal but expected):**
- `docs/models/<slug>/index.md` (slug hyphenated, id underscored),
  `docs/models/index.md` (table row), `README.md` (count + popular list +
  version table), `ComfyUI.md` (node catalog + control-image list),
  `example_workflows/rc_<id>_minimal.json`.

Nothing enumerates `ModelType` fatally except `src/api/v1/inference.py`
(eager). `pipeline_manager` and `cli/download_models` iterate it under
try/except.

---

## Phase 3 — Verify without a GPU (the gate before you ask for a machine)

`scripts/verify_registration.py` runs all of this. It must pass **with AND
without** an ai-toolkit checkout (`AI_TOOLKIT_PATH=/nonexistent`), because the
FastAPI startup path must survive a missing checkout.

1. Registry builds; model resolves through `get_pipeline_class` /
   `PIPELINE_REGISTRY`; `src.api.v1.inference` imports clean.
2. `PipelineConfig` values match the training defaults (steps/cfg/divisor).
3. ComfyUI: nodes registered, `RESOLUTION_STEP == resolution_divisor`,
   `ALL_PIPELINES` ⊇ ids AND every id in `ctor_map`.
4. Functional CPU test on a *tiny* build of the arch (see
   `tests/test_krea2_lora_merge.py`: ~0.7M-param SingleStreamDiT exercises the
   REAL upstream modules, no weights downloaded) — sampler runs, CFG path,
   seed determinism, the LoRA merge merges the expected module count with the
   right arithmetic, ref-latent + kv_cache paths.
5. `pytest tests/ -q` — no regressions.

Only when all pass do you spend a GPU.

---

## Phase 4 — Build a test image (branch, not tag)

The Dockerfile checks out **tags** for ai-toolkit but accepts a **branch** for
the inference repo. Build a test image off your feature branch:

```
cd Images/ai-toolkit-inference && source env.sh
./build.sh <AITOOLKIT_TAG_with_the_arch> <your-branch>
```

- `AITOOLKIT_TAG` must be ≥ the version that contains the arch. Krea 2 needs
  `v0.11.0-202607221`; a stale local ai-toolkit checkout (e.g. v0.7.21) does
  not have it.
- The image installs **ai-toolkit's** requirements (`transformers==5.5.3`,
  diffusers pin from `requirements_base.txt`) then only this repo's
  `requirements.txt` (web deps). This repo's `requirements-inference.txt`
  (`transformers==4.57.3`) is a LOCAL-dev file the image never installs — so
  version guards keyed to it (e.g. Krea 2 edit's Qwen3-VL M-RoPE check) pass
  in the container but can trip locally.
- Smoke-test the image before running anything: `docker run --rm <img>
  python3 -c "..."` to confirm the model is registered, the ai-toolkit arch
  files are present, transformers/diffusers have the needed classes.

**Set up the machine** with the `remote-experiment` skill (JuiceFS mount,
container, HF auth). VRAM: size from Phase 0 (Krea 2 ~35 GiB bf16 → 48 GB card
enough).

---

## Phase 5 — Parity-verify against REAL production LoRAs

Do not train your own — pull real user jobs from prod. They are the honest
test and they surface config variety you would never invent.

1. **Find jobs** in prod Supabase (`trainer_jobs`), status `stopped`, arch =
   yours. Prefer `quantize=false` + `quantize_te=false` jobs → clean bf16
   parity (the pipeline runs bf16; a float8-trained job will match on
   structure but drift at pixel level — that is expected, not a bug). For
   edit, you NEED jobs whose `sample.samples` carry `ctrl_img*` — most don't.
   `reference/find_prod_jobs.sql` has the queries.
   **Screen prompts for NSFW** before using a job as a test artifact.
2. **Pull assets** (all public via CDN once you have the job's user_id/name):
   `config.yaml`, the LoRA (`<name>_<step9>.safetensors`), control images
   (`.../ai-toolkit/data/images/...`). Only the **samples directory listing**
   needs the JuiceFS mount (filenames are `<ms>__<step9>_<idx>.jpg`,
   unpredictable; the CDN won't list a directory). Pair the LoRA checkpoint to
   the SAME step as the samples you compare against.
3. **Replay** through the running server: build the `/v1/inference` payload
   from `config.yaml` — same prompt / seed / steps / guidance / **per-prompt
   width×height** / neg. Respect `walk_seed` (false → all prompts share the
   seed). For edit, pass `kv_cache`/`match_target_res` from the job's
   `model_kwargs`, and the control-image URLs. Poll status; copy the output
   images out of the container's `/tmp/inference_output`.
4. **Compare**: `scripts/compare_samples.py --training-dir … --inference-dir …`
   → PSNR/SSIM + a training|inference|diff contact sheet. **Judge by eye, not
   by PSNR.** SSIM 0.9–0.99 + a mostly-black diff column = parity even at
   ~20 dB PSNR (two independent JPEG encodes + bf16 nondeterminism put a floor
   under PSNR). A real bug breaks composition, not one attribute.

---

## Phase 6 — Release (order is fixed; build is NOT first)

```
merge inference PR → main
  → add README version-table row (link the OSTRIS upstream commit that set
    that version.py, not our fork's merge commit) + tag v<aitk_ver>.<YYYYMMDD>n
  → ./build.sh <aitk_tag> <that inference tag>   ← build happens HERE
  → verify image in BOTH registries (DO + ECR), same digest
```

Building before the tag exists is meaningless — the Dockerfile pulls the
inference repo *by tag*.

### The two download gotchas — deployment WILL fail without pre-warming

- `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in current huggingface_hub; **xet
  is the only fast path** (`HF_XET_HIGH_PERFORMANCE=1`). With xet off, big
  single-file checkpoints crawl (~7.5 MB/s) and a cold first request blows the
  executor's 3600 s timeout.
- Freshly-granted **gated** repos throw `Unable to parse string as hex hash
  value` on xet's read-token the first time. The account must accept the
  licence on HF, and the cache should be pre-warmed out-of-band.

→ **Pre-warm the model cache before serving traffic.** Don't let request #1
download 26 GB.

---

## Phase 7 — Expose to users (separate repos — the model is invisible without this)

The backend supporting a model does NOT make it appear in the product.
`trainer-backend` has **no** inference code; the gate is in
`runcomfy-website` + the `trainer-contents` submodule.

**`trainer-contents` (PR against `main`), 3 files — `base_model` uses the
COLON arch name (`krea2:o_edit`), url_name uses hyphens:**
1. `base_model_and_machine_map.json` — feeds `INFERENCE_READY_BASE_MODELS`,
   which `isBaseModelInferenceReady()` / `isBaseModelDeploymentReady()`
   consult. **Without this the model shows no inference/deploy entry point.**
   Set `min_machine` from Phase 0 VRAM.
2. `inference_pricing.json` — per-request price + `url_name`. `base_price` is
   thousandths of USD per megapixel (formula `(width*height)/1000000`, ×
   `num_frames` for video). id continues the `…60NN` sequence.
3. `inference_schemas_with_lora/<file_name>.json` — the request FORM. **Every
   pricing row's `file_name` must resolve to one of these** (missed on the
   first Krea 2 PR — form rendered nothing). Mirror `flux2.json` for
   multi-reference edit, `zimage_turbo.json` for plain t2i. **Cross-check every
   default against the shipped `PipelineConfig`, and feed min/max through the
   real `PromptItem`/`InferenceInput` validators** — a form value the server
   422s is a broken form.

**`runcomfy-website` (PR against `develop`):** bump the `trainer-contents`
submodule pin. Combine with any other submodule bump into ONE PR — each PR is
an expensive Vercel build. Verify the app reads the new data through the pin
before pushing.

### Pricing guidance
- Anchor to the market (fal etc.) for the **same open weights**, not a
  vendor's closed hosted variant.
- Don't price up for our slowness (unoptimized bf16, no compile) — that's an
  engineering task, not the customer's problem. Sanity-check revenue-per-
  GPU-second against the catalogue, don't set price *from* it.
- Edit variants run the same weights as their base → same price (precedent:
  `wan22_14b_i2v` == `wan22_14b:t2v`).
- Keep numbers as clean integers matching existing tiers.

---

## Scripts in this skill
- `scripts/verify_registration.py` — Phase 3 CPU gate (run with and without
  `AI_TOOLKIT_PATH`).
- `scripts/compare_samples.py` — Phase 5 parity report + contact sheet.
- `reference/find_prod_jobs.sql` — Phase 5 prod-job discovery queries.
- `reference/pricing_field_reference.md` — the two trainer-contents JSON
  schemas, field by field.
- `reference/how-krea2-was-analyzed.md` — the Phase 0 research approach.
