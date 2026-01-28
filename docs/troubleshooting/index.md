---
title: "Troubleshooting AI Toolkit LoRA Inference (Preview vs Inference Mismatch)"
description: "Fix the common problems when running LoRAs trained with ostris/ai-toolkit: mismatch between training samples and Diffusers/ComfyUI inference, LoRA not applying, control-image wiring, Wan2.2 MoE LoRA format, and Hugging Face download issues."
keywords: "ai-toolkit lora inference troubleshooting, preview vs inference mismatch, comfyui vs ai toolkit, diffusers lora not working, wan2.2 lora low high noise, huggingface download stuck 99, flex2 diffusers pipeline"
permalink: /troubleshooting/
---

← [Docs Home](../) · [Model Catalog](../models/) · [HTTP API](../api/)

# Troubleshooting AI Toolkit LoRA Inference

This page is a practical checklist for the most common “**my AI Toolkit samples look great, but inference looks different / broken**” problems.

It’s written specifically for:

- LoRAs trained with **`ostris/ai-toolkit`**
- Inference in **Diffusers / ComfyUI / custom stacks**
- Using **ai-toolkit-inference** (this repo) as the “known-good” reference implementation

If you want a known-good baseline, start by running the corresponding **by-model pipeline** from this repo, then change one variable at a time.

## Quick navigation

- [One-minute checklist](#one-minute-checklist)
- [Symptom table](#symptom-cause-fix)
- [Training samples better than inference](#training-samples-better-than-inference)
- [LoRA not applying / no effect](#lora-not-applying)
- [API 400 errors](#api-400-errors)
- [Wan video issues](#wan-video-issues)
- [Hugging Face download stalls](#hf-download-stalls)

---

<a id="one-minute-checklist"></a>

## One-minute checklist (80/20)

When outputs drift from training samples, it’s almost always one of these:

1) **Wrong model id / wrong base checkpoint**
- Confirm the **exact** model you trained against (e.g. FLUX vs FLUX.2 vs Flex.2, Wan 2.1 480P vs 720P).
- Use the [Model Catalog](../models/) and compare the base checkpoint.

2) **Resolution rules don’t match**
- This server **snaps width/height down** to a multiple of `resolution_divisor` (per model).
- If your training used buckets/crops, “same numbers” in UI can still land on different internal sizes.

3) **Scheduler / steps / guidance differ**
- Distilled / few-step models are extremely sensitive (e.g. Z-Image Turbo).
- Some models use non-standard guidance params (e.g. Qwen uses `true_cfg_scale`).

4) **Control image wiring differs**
- Edit/I2V models often resize or encode prompts *using the image*.
- A different resize/crop policy is a different conditioning signal.

5) **LoRA isn’t being applied the way you think**
- “Adapters” vs `fuse_lora` vs model-specific transformer merges.
- Wrong trigger word, wrong scale, incompatible LoRA key layout.

---

<a id="symptom-cause-fix"></a>

## Symptom → cause → fix (quick table)

| Symptom | Most likely cause | What to do in this repo |
|---|---|---|
| AI Toolkit samples look good, Diffusers/ComfyUI inference looks different | different base checkpoint / scheduler / resolution snapping | Start from the **model page** and copy its defaults; also check snapped dimensions and LoRA load mode. |
| LoRA “does nothing” (looks like base model) | wrong trigger word / scale 0 / incompatible LoRA keys / wrong pipeline family | Verify `trigger_word`, set `loras[].network_multiplier=1.0`, and use the exact model pipeline. For some families, LoRA is **fused** so scale is not dynamically adjustable. |
| API returns `CONTROL_IMAGE_REQUIRED` | edit/I2V model needs `ctrl_img` | Provide `ctrl_img` (URL or base64) in the prompt item. |
| API returns `MOE_FORMAT_REQUIRED`, `SINGLE_MOE_CONFIG_ONLY`, or `SINGLE_LORA_ONLY` | Wan 2.2 MoE format required (or multiple configs sent), or multiple LoRAs sent | Use `loras` with `transformer: "low"` / `"high"` for Wan 2.2 14B (single config); otherwise send exactly one LoRA item. |
| Wan2.2 motion is “too fast / weird” vs samples | frames/fps mismatch; I2V conditioning mismatch | Match `num_frames` + `fps` defaults from the model page; keep resolution within the checkpoint regime. |
| Download stalls at ~99% | HF transfer/Xet edge cases | See the Hugging Face download section below; this repo already applies mitigations. |
| OOM / CUDA out of memory | too-large res/frames, heavy pipeline, CPU offload disabled for that model | Reduce `width/height/num_frames`, try smaller model ids, enable CPU offload where supported. |

---

<a id="training-samples-better-than-inference"></a>

## 1) “Training samples are better than my inference”

This is a very common report across model families (FLUX/HiDream/Qwen/Wan). Users often find that:

- AI Toolkit’s **sample images/videos** look good
- Running the same prompt/seed in another stack looks “washed out / different / worse”

Examples (for context):
- AI Toolkit issue: “training samples are much better than inference in ComfyUI” ([#309](https://github.com/ostris/ai-toolkit/issues/309))
- Reddit threads discussing FLUX LoRA preview vs inference drift (search for “AI Toolkit FLUX LoRA inference different”) 

What usually fixes it:

- Start from the relevant **model page** in this repo and match:
  - **base checkpoint**
  - **scheduler config** (some pipelines in this repo intentionally pin scheduler params)
  - **steps & guidance**
  - **snapped resolution** (`width//divisor*divisor`)
  - **LoRA load mode** (adapter vs fused)

If you’re not sure which model you are actually using, call `GET /v1/models` (see [HTTP API](../api/)) and compare defaults.

---

<a id="lora-not-applying"></a>

## 2) “My LoRA isn’t applying / has no effect” (Diffusers / ComfyUI)

This shows up as “the output looks exactly like the base model.” Common causes:

### A) Missing trigger word

AI Toolkit commonly uses a **trigger token** that you must include in the inference prompt.

This server supports a convenience placeholder:

- Put `[trigger]` in your prompt
- Set `trigger_word: "..."` at the request level

The server replaces `[trigger]` in every prompt item (see executor code in `src/tasks/executor.py`).

### B) LoRA scale is too low / not actually being updated

- For “adapter-style” pipelines (most), `loras[].network_multiplier` changes LoRA strength dynamically via `set_lora_scale()`.
- For “fused” pipelines (e.g. `fuse_lora`), the LoRA is merged at load time, so changing `loras[].network_multiplier` requires a reload.

The API uses a **single LoRA scale per request** (per transformer for MoE). If you need multiple scales, send separate requests.

### C) Incompatible LoRA key layout (especially Qwen Image)

Some users report Qwen-Image LoRAs not working in other stacks due to weight key naming expectations.

Example report: “Qwen-Image LoRAs not working in ComfyUI” ([ai-toolkit #372](https://github.com/ostris/ai-toolkit/issues/372)).

If your LoRA works in AI Toolkit samples but not elsewhere:

- Use the **Qwen Image model pipeline** from this repo (it passes guidance as `true_cfg_scale` and aligns prompt/image encoding with AI Toolkit).
- If you must use another stack, you may need a **LoRA conversion/renaming** step.

### D) Diffusers version / pipeline compatibility

Some model families require newer or customized Diffusers behavior for LoRA injection.

Example: LoRA not affecting FLUX inference due to compatibility issues was discussed in Diffusers ([diffusers #9361](https://github.com/huggingface/diffusers/issues/9361)).

This repo pins Diffusers to a specific revision in `requirements-inference.txt` to reduce “works in one environment but not another” failures.

---

<a id="api-400-errors"></a>

## 3) “I’m getting a 400 error from the API”

Most 400s are intentional guardrails.

### A) `CONTROL_IMAGE_REQUIRED`

Models that require a control image:

- edit models (e.g. FLUX Kontext, Qwen Image Edit)
- I2V video models (Wan I2V, Wan2.2 I2V)

Fix:

- Provide `ctrl_img` (or `ctrl_img_1..3` for multi-image models) inside each prompt item.
- Use a URL or base64 string. See [HTTP API](../api/) for formats.

### B) `MOE_FORMAT_REQUIRED` (Wan 2.2 14B)

Wan 2.2 14B uses a **dual-transformer MoE setup**. This server expects:

```json
"loras": [
  {"path": "low_noise.safetensors", "transformer": "low", "network_multiplier": 1.0},
  {"path": "high_noise.safetensors", "transformer": "high", "network_multiplier": 1.0}
]
```

Context: users frequently ask about the “low/high noise” split and when each matters (e.g. [ai-toolkit #349](https://github.com/ostris/ai-toolkit/issues/349)).

Practical tip:

- If you only trained one side, you can supply just one key (`low` or `high`).
- Current implementation supports **one** MoE config per request (`SINGLE_MOE_CONFIG_ONLY`).

---

<a id="wan-video-issues"></a>

## 4) Wan video looks wrong (frames/fps/resolution)

### Wan 2.1 (T2V/I2V)

- Match the checkpoint variant (480P vs 720P) and keep resolution in that regime.
- Use dimensions divisible by the model’s divisor (often 16 or 32 depending on the pipeline).

### Wan 2.2 I2V “motion too fast / weird”

Users have reported Wan2.2 I2V training/inference oddities, including motion artifacts that can correlate with frame/time settings (see [ai-toolkit #421](https://github.com/ostris/ai-toolkit/issues/421)).

What to try:

- Start from the model page defaults: `num_frames=41`, `fps=16`, steps/guidance defaults.
- Avoid changing multiple knobs at once.

---

## 5) Flex.2 / custom pipelines don’t run in vanilla Diffusers

Some preview models ship with custom or rapidly changing pipeline code.

Example: Flex.2 users discuss Diffusers incompatibilities and the need for custom pipelines ([Flex.2 discussion](https://huggingface.co/ostris/Flex.2-preview/discussions/1)).

Fix:

- Use this repo’s **Flex.2** pipeline (it uses a custom pipeline path and pinned dependencies).

---

<a id="hf-download-stalls"></a>

## 6) Hugging Face model download stalls at ~99%

This is a real-world issue when downloading large model repos.

Example reports:

- `hf_transfer` download stuck at 99% ([hf_transfer #30](https://github.com/huggingface/hf_transfer/issues/30))
- `huggingface_hub` snapshot_download stalls ([huggingface_hub #2197](https://github.com/huggingface/huggingface_hub/issues/2197))

This repo includes mitigations in `src/services/pipeline_manager.py` (environment vars set before `snapshot_download`). If you still hit stalls:

- Try disabling `hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER`
- Ensure you have enough disk space
- Set `HF_TOKEN` / pass `hf_token` if the model is gated

---

## 7) “AI Toolkit expects train_config during inference”

Some users assume the training config must be present to run inference (see [ai-toolkit #416](https://github.com/ostris/ai-toolkit/issues/416)).

This server does **not** read the AI Toolkit training config to infer parameters automatically. Instead:

- you pass inference parameters explicitly (steps/guidance/resolution/etc.)
- the model pages show the best-known defaults for preview matching

---

## Still stuck?

If you can share:

- the **model id** (`GET /v1/models`)
- the request JSON (without base64 blobs)
- the LoRA filename(s)
- the snapped output resolution

…you can usually pinpoint the mismatch quickly.

Note: if your main blocker is **environment drift** (CUDA/PyTorch/Diffusers versions, or missing model-specific pipeline deps), it can help to run training + inference in a fixed runtime/container. RunComfy provides a managed runtime for AI Toolkit, but any reproducible GPU environment works — the reference behavior is still defined by the code in this repo.
