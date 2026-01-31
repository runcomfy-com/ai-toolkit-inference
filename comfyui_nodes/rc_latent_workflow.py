"""Latent-workflow ComfyUI nodes for ai-toolkit-inference.

These nodes are designed to mimic the common SD/SDXL ComfyUI pattern:
  EmptyLatent -> Sampler -> LatentUpscale -> Sampler(denoise<1) -> Decode

For Qwen Image models, the underlying diffusers pipeline uses *packed* latents internally.
The ai-toolkit-inference Qwen wrappers convert them to standard spatial latents
[B, C, H/8, W/8] so ComfyUI's built-in latent ops (LatentUpscale, etc.) work.

Node categories use "RunComfy-Inference/Workflow" for consistency with ComfyUI.md.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .rc_common import get_or_load_pipeline, comfy_to_pil_image, pil_frames_to_comfy_images

# Consistent category for all latent workflow nodes
WORKFLOW_CATEGORY = "RunComfy-Inference/Workflow"


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


def _comfy_batch_to_tensor(img: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI IMAGE [B,H,W,C] in [0,1] to model tensor [B,C,H,W] in [-1,1]."""
    # ComfyUI: [B, H, W, C] float32 in [0,1]
    # Model: [B, C, H, W] in [-1,1]
    t = img.permute(0, 3, 1, 2)  # [B,C,H,W]
    t = t * 2.0 - 1.0  # [0,1] -> [-1,1]
    return t


def _tensor_to_comfy_batch(t: torch.Tensor) -> torch.Tensor:
    """Convert model tensor [B,C,H,W] in [0,1] to ComfyUI IMAGE [B,H,W,C]."""
    # Model output: [B, C, H, W] in [0,1]
    # ComfyUI: [B, H, W, C] float32 in [0,1]
    return t.permute(0, 2, 3, 1).float().cpu()


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
    CATEGORY = WORKFLOW_CATEGORY

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
    CATEGORY = WORKFLOW_CATEGORY

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


class RCAITKEmptyLatent:
    """Create an empty latent for use with RCAITKSampler.
    
    This mirrors ComfyUI's EmptyLatentImage node for a native workflow:
    EmptyLatent -> Sampler -> LatentUpscale -> Sampler(denoise<1) -> Decode
    
    The output LATENT has a special marker so the sampler knows to do txt2img
    instead of using the latent as img2img input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 16,
                        "tooltip": "Image width (will be snapped to model's resolution divisor)",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 16,
                        "tooltip": "Image height (will be snapped to model's resolution divisor)",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of images to generate in parallel",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "create"
    CATEGORY = WORKFLOW_CATEGORY

    def create(self, pipe, width: int, height: int, batch_size: int):
        # Get latent dimensions from pipeline config
        vae_sf = int(getattr(getattr(pipe, "pipe", None), "vae_scale_factor", 8))
        divisor = int(getattr(getattr(pipe, "CONFIG", None), "resolution_divisor", 16))
        
        # Snap dimensions to divisor
        width = (width // divisor) * divisor
        height = (height // divisor) * divisor
        
        # Calculate latent size
        lat_h = height // vae_sf
        lat_w = width // vae_sf
        
        # Determine latent channels (4 for SD/SDXL, 16 for Qwen)
        # Try to get from VAE config, default to 4
        lat_channels = 4
        try:
            vae = getattr(pipe.pipe, "vae", None)
            if vae is not None:
                lat_channels = getattr(vae.config, "latent_channels", 4)
        except Exception:
            pass
        
        # Create empty latent (zeros)
        samples = torch.zeros(
            (batch_size, lat_channels, lat_h, lat_w),
            dtype=torch.float32,
        )
        
        # Return with marker indicating this is an "empty" latent for txt2img
        return ({
            "samples": samples,
            "aitk_empty": True,
            "aitk_width": width,
            "aitk_height": height,
        },)


class RCAITKSampler:
    """Sample latents using a ComfyUI-native interface.
    
    This node requires a LATENT input (use RCAITKEmptyLatent for txt2img).
    Width/height are inferred from the latent shape.
    
    When the input is from RCAITKEmptyLatent and denoise=1.0, this performs txt2img.
    When the input has actual latent data or denoise<1.0, this performs img2img/refine.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "latent": ("LATENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
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
                        "tooltip": "1.0 = full generation (txt2img), <1.0 = refinement (img2img)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = WORKFLOW_CATEGORY

    def sample(
        self,
        pipe,
        latent,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg: float,
        seed: int,
        denoise: float,
    ):
        samples = latent.get("samples", None)
        if samples is None:
            raise ValueError("LATENT input is missing 'samples'")

        is_empty = latent.get("aitk_empty", False)
        
        # Get dimensions from latent
        vae_sf = int(getattr(getattr(pipe, "pipe", None), "vae_scale_factor", 8))
        
        # Use stored dimensions if available (from EmptyLatent), else compute from shape
        if "aitk_width" in latent and "aitk_height" in latent:
            width = latent["aitk_width"]
            height = latent["aitk_height"]
        else:
            height = int(samples.shape[-2] * vae_sf)
            width = int(samples.shape[-1] * vae_sf)

        # Determine if we should do txt2img or img2img
        # txt2img: empty latent + denoise=1.0
        # img2img: real latent data OR denoise<1.0
        do_txt2img = is_empty and denoise >= 1.0

        if do_txt2img:
            # Full txt2img: don't pass latents to pipeline
            result = pipe.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                seed=int(seed),
                output_type="latent",
                latents=None,
                denoise_strength=1.0,
            )
        else:
            # img2img/refine: pass latents to pipeline
            result = pipe.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                seed=int(seed),
                output_type="latent",
                latents=samples,
                denoise_strength=float(denoise),
            )

        out_latents = result.get("latents", None)
        if out_latents is None:
            raise ValueError("Pipeline did not return 'latents'")

        # Return clean latent dict (no markers)
        return ({"samples": out_latents},)


class RCAITKDecodeLatent:
    """Decode LATENT -> IMAGE using the pipeline's VAE.
    
    Uses batch decode when available for better performance.
    """

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
    CATEGORY = WORKFLOW_CATEGORY

    def decode(self, pipe, latent):
        lat = latent.get("samples", None)
        if lat is None:
            raise ValueError("LATENT input is missing 'samples'")

        # Use batch decode if available (more efficient for batches)
        if hasattr(pipe, "decode_latents_to_images"):
            # Batch decode returns [B,C,H,W] in [0,1]
            images = pipe.decode_latents_to_images(lat)
            return (_tensor_to_comfy_batch(images),)

        # Fallback: decode each batch item separately
        frames = []
        for i in range(lat.shape[0]):
            pil = pipe.decode_latent_to_image(lat[i : i + 1])
            frames.append(pil)

        return (pil_frames_to_comfy_images(frames),)


class RCAITKEncodeImage:
    """Encode IMAGE -> LATENT using the pipeline's VAE.
    
    Uses batch encode when available for better performance.
    """

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
    CATEGORY = WORKFLOW_CATEGORY

    def encode(self, pipe, image):
        # Use batch encode if available (more efficient for batches)
        if hasattr(pipe, "encode_images_to_latents"):
            # Convert ComfyUI [B,H,W,C] in [0,1] to model [B,C,H,W] in [-1,1]
            img_tensor = _comfy_batch_to_tensor(image)
            latents = pipe.encode_images_to_latents(img_tensor)
            return ({"samples": latents},)

        # Fallback: encode each image separately
        pil_list = _comfy_batch_to_pil_list(image)
        latents = []
        for pil in pil_list:
            lat = pipe.encode_image_to_latent(pil)
            latents.append(lat)

        out = torch.cat(latents, dim=0)
        return ({"samples": out},)
