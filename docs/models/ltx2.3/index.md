---
title: "LTX-2.3 LoRA Inference with Diffusers (AI Toolkit-trained)"
description: "Run ltx2.3 LoRAs trained with ostris/ai-toolkit using the ai-toolkit-inference reference pipeline."
keywords: "ltx2.3 lora inference, LTX-2.3 diffusers pipeline, ai-toolkit ltx2.3 inference, ltx2.3"
permalink: /models/ltx2.3/
---

<- [Docs Home](../../) . [Model Catalog](../) . [HTTP API](../../api/) . [Troubleshooting](../../troubleshooting/)
# LTX-2.3 LoRA Inference with Diffusers (AI Toolkit-trained)

**API model id:** `ltx2.3`
**URL slug:** `ltx2.3`

This page documents the **reference Diffusers inference pipeline** for `ltx2.3` (LTX-2.3). It inherits all behavior from the [LTX-2 pipeline](../ltx2/) with the following differences:

- **Base model:** `dg845/LTX-2.3-Diffusers` (community-converted diffusers format; official Lightricks repo `Lightricks/LTX-2.3` is single-file checkpoint only)
- **Vocoder:** Uses `LTX2VocoderWithBWE` (Bandwidth Extension) for improved audio quality

## Quick facts

| Field | Value |
|---|---|
| Pipeline | [`src/pipelines/ltx2.py` (LTX23Pipeline)](https://github.com/runcomfy-com/ai-toolkit-inference/blob/main/src/pipelines/ltx2.py) |
| Base checkpoint | `dg845/LTX-2.3-Diffusers` |
| Defaults | `sample_steps=25`, `guidance_scale=4.0`, `seed=42` |
| Resolution snapping | Floors width/height to a multiple of **32** |
| Control image | Optional (`ctrl_img` switches to I2V) |
| Video | Yes (`num_frames=41`, `fps=24` by default) |
| LoRA scale behavior | Same as LTX-2: LoRA converted (AI Toolkit -> diffusers) and applied via `set_adapters` (hotswap). |
| Needs AI Toolkit | Optional (recommended for LoRA conversion helpers via `AI_TOOLKIT_PATH`) |

## Minimal API request

```json
{
  "model": "ltx2.3",
  "trigger_word": "sks",
  "prompts": [
    {
      "prompt": "[trigger] a photo of a person",
      "width": 768,
      "height": 512,
      "seed": 42,
      "sample_steps": 25,
      "guidance_scale": 4.0,
      "neg": "",
      "num_frames": 41,
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

## Pipeline behavior

All behavior is inherited from the LTX-2 pipeline. See the [LTX-2 documentation](../ltx2/) for details on:
- T2V and I2V modes
- LoRA loading and conversion
- Preview-matching notes
- Debugging mismatch

## Related

- [LTX-2](../ltx2/)
