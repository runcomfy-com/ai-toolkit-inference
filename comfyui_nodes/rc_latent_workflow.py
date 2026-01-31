"""Latent-workflow ComfyUI nodes for ai-toolkit-inference.

This file was missing from the clean bundle.

These nodes are designed to mimic the common SD/SDXL ComfyUI pattern:
  text2img -> (LATENT) -> LatentUpscale -> refine (denoise<1) -> decode

For Qwen Image models, the underlying diffusers pipeline uses *packed* latents internally.
The ai-toolkit-inference Qwen wrappers convert them to standard spatial latents
[B, C, H/8, W/8] so ComfyUI's built-in latent ops (LatentUpscale, etc.) work.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .rc_common import get_or_load_pipeline, comfy_to_pil_image, pil_frames_to_comfy_images


def _comfy_batch_to_pil_list(img: torch.Tensor) -> list[Image.Image]:
    """Convert a ComfyUI IMAGE tensor [B,H,W,C] in 0..1 to a list of PIL images."""
    if img.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(img.shape)}")

    out: list[Image.Image] = []
    for i in range(img.shape[0]):
        arr = img[i].detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out


class RCAITKLoRA:
    """Helper node: select a LoRA file from ComfyUI's loras folder."""

    @classmethod
    def INPUT_TYPES(cls):
        # Import inside to avoid hard dependency outside ComfyUI.
        import folder_paths

        loras = folder_paths.get_filename_list("loras")
        # If no LoRAs are present, ComfyUI still expects a list for the dropdown.
        if not loras:
            loras = [""]

        return {
            "required": {
                "lora_name": (loras, {"tooltip": "Pick a LoRA from ComfyUI models/loras"}),
                "lora_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "LoRA strength (applied when loading the pipeline)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AITK_LORA",)
    FUNCTION = "build"
    CATEGORY = "RunComfy/ai-toolkit/latent"

    def build(self, lora_name: str, lora_scale: float):
        # Keep it simple: a dict with paths + scale.
        # The loader node will resolve the full path.
        return ({"lora_name": lora_name, "lora_scale": float(lora_scale)},)


class RCAITKLoadPipeline:
    """Load an ai-toolkit pipeline for latent workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (
                    [
                        "sd15",
                        "sdxl",
                        "qwen_image",
                        "qwen_image_2512",
                    ],
                    {"tooltip": "Which base model pipeline to load"},
                ),
                "enable_cpu_offload": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable diffusers model CPU offload (lower VRAM, slower)",
                    },
                ),
                "low_ram_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use sequential CPU offload for systems with <64GB RAM (much slower but uses less RAM)",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional Hugging Face token for gated/private repos",
                    },
                ),
            },
            "optional": {
                "lora": ("AITK_LORA",),
            },
        }

    RETURN_TYPES = ("AITK_PIPELINE",)
    FUNCTION = "load"
    CATEGORY = "RunComfy/ai-toolkit/latent"

    def load(self, pipeline: str, enable_cpu_offload: bool, low_ram_mode: bool, hf_token: str, lora=None):
        # Import here so ComfyUI can load the node pack even if some deps aren't installed.
        from src.pipelines.sd15 import SD15Pipeline
        from src.pipelines.sdxl import SDXLPipeline
        from src.pipelines.qwen_image import QwenImagePipeline, QwenImage2512Pipeline

        ctor_map = {
            "sd15": SD15Pipeline,
            "sdxl": SDXLPipeline,
            "qwen_image": QwenImagePipeline,
            "qwen_image_2512": QwenImage2512Pipeline,
        }

        pipeline_ctor = ctor_map[pipeline]

        lora_paths = []
        lora_scale = 1.0

        if lora is not None:
            import folder_paths

            name = (lora.get("lora_name") or "").strip()
            if name:
                lora_paths = [folder_paths.get_full_path_or_raise("loras", name)]
            lora_scale = float(lora.get("lora_scale", 1.0))

        # Empty token -> None
        token = hf_token.strip() or None

        pipe = get_or_load_pipeline(
            model_id=pipeline,
            pipeline_ctor=pipeline_ctor,
            lora_paths=lora_paths,
            lora_scale=lora_scale,
            enable_cpu_offload=bool(enable_cpu_offload),
            hf_token=token,
        )

        # Enable sequential CPU offload for low RAM systems (<64GB)
        # This is slower but uses much less RAM
        if low_ram_mode and enable_cpu_offload:
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()

        return (pipe,)


class RCAITKSampler:
    """Sample latents (and optionally refine from incoming latents)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 16,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 16,
                    },
                ),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "1.0 = full generation, <1.0 = refinement strength",
                    },
                ),
            },
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "RunComfy/ai-toolkit/latent"

    def sample(
        self,
        pipe,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        denoise: float,
        latent=None,
    ):
        latents_in = None
        if latent is not None:
            latents_in = latent.get("samples", None)
            if latents_in is None:
                raise ValueError("LATENT input is missing 'samples'")

            # If an input latent is provided, infer the correct output resolution from the latent shape.
            vae_sf = int(getattr(getattr(pipe, "pipe", None), "vae_scale_factor", 8))
            height = int(latents_in.shape[-2] * vae_sf)
            width = int(latents_in.shape[-1] * vae_sf)

        result = pipe.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            seed=int(seed),
            output_type="latent",
            latents=latents_in,
            denoise_strength=float(denoise),
        )

        out_latents = result.get("latents", None)
        if out_latents is None:
            raise ValueError("Pipeline did not return 'latents'")

        return ({"samples": out_latents},)


class RCAITKDecodeLatent:
    """Decode LATENT -> IMAGE using the pipeline's VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "RunComfy/ai-toolkit/latent"

    def decode(self, pipe, latent):
        lat = latent.get("samples", None)
        if lat is None:
            raise ValueError("LATENT input is missing 'samples'")

        # Decode each batch item as a frame.
        frames = []
        for i in range(lat.shape[0]):
            pil = pipe.decode_latent_to_image(lat[i : i + 1])
            frames.append(pil)

        return (pil_frames_to_comfy_images(frames),)


class RCAITKEncodeImage:
    """Encode IMAGE -> LATENT using the pipeline's VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "RunComfy/ai-toolkit/latent"

    def encode(self, pipe, image):
        pil_list = _comfy_batch_to_pil_list(image)

        latents = []
        for pil in pil_list:
            lat = pipe.encode_image_to_latent(pil)
            latents.append(lat)

        out = torch.cat(latents, dim=0)
        return ({"samples": out},)
