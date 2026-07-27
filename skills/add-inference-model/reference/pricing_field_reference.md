# trainer-contents pricing/exposure fields (Phase 7)

Three files in `InceptionsAI/trainer-contents` gate a model in the website.
PR against `main`. `base_model` uses the **colon** arch name; `url_name` and
`api_slug` use **hyphens**.

## 1. `base_model_and_machine_map.json`

Feeds `INFERENCE_READY_BASE_MODELS` → `isBaseModelInferenceReady()` /
`isBaseModelDeploymentReady()`. **Without an entry the model has no inference
or deployment entry point in the UI**, regardless of backend support.

```json
{
  "base_model": "krea2:o_edit",
  "api_slug": "runcomfy/krea2-o-edit",          // display only
  "min_machine": "48G (A6000)",                  // from Phase 0 peak VRAM
  "machines": ["48G (A6000)", "48G Plus (L40S, L40)", "80G (A100)",
               "80G Plus (H100)", "141G (H200)"]
}
```

## 2. `inference_pricing.json`

Per-request price + the `url_name` behind `BASE_MODEL_TO_URL_NAME`.
`base_price` = thousandths of USD per megapixel. id continues the `…60NN`
sequence. `file_name` MUST resolve to a schema in #3.

```json
{
  "id": "00000000-0000-0000-0000-000000006016",
  "file_name": "krea2_o_edit.json",
  "base_model": "krea2:o_edit",
  "url_name": "krea2-o-edit",
  "list_name": "Krea 2 Edit",
  "list_credits_remarks": "$0.045 per megapixel.",
  "credits_remarks": "Your request will cost $0.045 per megapixel of generated image.",
  "task_templates": [{
    "base_price": 45,
    "price_alpha": { "width": {}, "height": {} },
    "price_alpha_formula": { "multiplier": "(width*height)/1000000" }
  }]
}
```

Video multiplies by frames: `"(width*height)/1000000*num_frames"`.

## 3. `inference_schemas_with_lora/<file_name>.json`

The inference request FORM (OpenAPI-ish). Missing this = model shows as
inference-ready but renders no parameters. Mirror `zimage_turbo.json` (t2i) or
`flux2.json` (multi-reference edit). Keys: `LoRAInput`, `Input`, `Output`.

`Input.properties` fields + formats used:
- `loras` array (maxItems 1), `prompt` (`str`, maxLength 2000)
- `ctrl_img_1..3` (`image_uri`) for edit; put `ctrl_img_1` in `required`
- `width`/`height` (`int_slider_with_range`, min 64 max 4096, **`multipleOf`
  = resolution_divisor**), default from PipelineConfig
- `guidance_scale` (`float_slider_with_range` 0–20), `sample_steps`
  (`int_slider_with_range`, max = a sane per-arch ceiling)
- `neg` (`str`), `seed` (`int_with_arrows_and_random` 0–2147483647),
  `sampler` (`str_with_choice`, enum from the arch)
- per-request options as `boolean`/`bool` (e.g. `kv_cache`,
  `match_target_res`) with a description telling users when to flip them
- `num_frames`/`fps` present but pinned to 1 for image models

**Verify before PR:**
1. every default matches the shipped `PipelineConfig`;
2. every field name maps onto `PromptItem` / `InferenceInput` (unknown fields
   are silently ignored);
3. the schema's default/min/max values all pass the real Pydantic validators
   (a value the form allows but the server 422s is a broken form);
4. `required` control-image matches `requires_control_image`.
