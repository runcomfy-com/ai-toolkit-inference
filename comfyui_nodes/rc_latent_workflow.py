"""Latent-workflow ComfyUI nodes for ai-toolkit-inference.

These nodes are designed to mimic the common SD/SDXL ComfyUI pattern:
  EmptyLatent -> Sampler -> LatentUpscale -> Sampler(denoise<1) -> Decode

For Qwen Image models, the underlying diffusers pipeline uses *packed* latents internally.
The ai-toolkit-inference Qwen wrappers convert them to standard spatial latents
[B, C, H/8, W/8] so ComfyUI's built-in latent ops (LatentUpscale, etc.) work.

Node categories use "RunComfy-Inference/Workflow" for consistency with ComfyUI.md.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from .rc_common import (
    comfy_to_pil_image,
    get_or_load_pipeline,
    pil_frames_to_comfy_images,
    pil_to_comfy_image,
    video_tensor_to_comfy_images,
)

logger = logging.getLogger(__name__)

# Consistent category for all latent workflow nodes
WORKFLOW_CATEGORY = "RunComfy-Inference/Workflow"

# Sentinel for the node's fps input: 0 means "whatever this pipeline generates
# at". It has to be a value no user would ever type, because the alternative --
# treating some real fps as "untouched" -- makes that fps unselectable. Picking
# 16 for that role meant an explicit 16 on LTX-2 (default_fps=24) came out as 24.
_FPS_USE_MODEL_DEFAULT = 0


def _resolve_fps(requested: int, model_fps: int | None) -> int:
    """Resolve the node's fps input against the pipeline's own timeline.

    A model that generates on a fixed timeline knows its own fps: H3 produces
    24 fps with audio muxed to match and rejects anything else, so a generic
    node-wide default would fail every stock workflow. The sentinel asks for
    that model default; anything else is the user's explicit choice and is
    passed through untouched, including values the model will refuse.
    """
    if int(requested) == _FPS_USE_MODEL_DEFAULT and model_fps:
        return int(model_fps)
    return int(requested)


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


def _latent_metadata_from_pipe(pipe, samples=None, *, width=None, height=None) -> dict:
    """Best-effort latent metadata (non-fatal if unavailable)."""
    if hasattr(pipe, "latent_metadata") and callable(getattr(pipe, "latent_metadata")):
        try:
            return pipe.latent_metadata(samples, width=width, height=height)
        except Exception:
            return {}
    return {}


def _validate_latent_compat(pipe, latent: dict) -> None:
    """Validate latent metadata against the target pipeline (if metadata is present)."""
    if not isinstance(latent, dict):
        return

    meta_type = latent.get("aitk_model_type")
    cfg = getattr(pipe, "CONFIG", None)
    pipe_type = getattr(cfg, "model_type", None)
    pipe_type_val = getattr(pipe_type, "value", None) if pipe_type is not None else None

    if meta_type and pipe_type_val and meta_type != pipe_type_val:
        raise ValueError(f"Latent model_type mismatch: latent={meta_type}, pipe={pipe_type_val}")


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
                "lora_path_or_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional override: local filesystem path or URL to a LoRA. If set, this overrides the dropdown.",
                    },
                ),
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

    def build(self, lora_name: str, lora_path_or_url: str, lora_scale: float):
        # Keep it simple: a dict with paths + scale.
        # The loader node will resolve the full path.
        override = (lora_path_or_url or "").strip()
        chosen = override if override else (lora_name or "").strip()
        return (
            {
                "lora_name": chosen,
                "lora_path_or_url": override,
                "lora_scale": float(lora_scale),
            },
        )


class RCAITKLoadPipeline:
    """Load an ai-toolkit pipeline for modular workflows.
    
    Supports all ai-toolkit pipelines. For SD15, SDXL, and Qwen Image models,
    you can use the full latent workflow (EmptyLatent -> Sampler -> Decode).
    For other models, use the RCAITKGenerate node for inference.
    """

    # Complete list of all supported pipelines
    ALL_PIPELINES = [
        # Latent workflow supported
        "sd15",
        "sdxl",
        "qwen_image",
        "qwen_image_2512",
        # Image-only (use RCAITKGenerate)
        "flux",
        "flux_kontext",
        "flux2",
        "flux2_klein_4b",
        "flux2_klein_9b",
        "flex1",
        "flex2",
        "zimage",
        "zimage_turbo",
        "zimage_deturbo",
        "chroma",
        "hidream",
        "hidream_e1",
        "lumina2",
        "omnigen2",
        "qwen_image_edit",
        "qwen_image_edit_plus",
        "qwen_image_edit_plus_2511",
        "krea2",
        "krea2_turbo",
        "krea2_o_edit",
        "krea2_o_edit_turbo",
        "minimax_h3",
        # Video models
        "ltx2",
        "ltx2.3",
        "wan21_14b",
        "wan21_1b",
        "wan21_i2v_14b",
        "wan21_i2v_14b480p",
        "wan22_14b_t2v",
        "wan22_14b_i2v",
        "wan22_5b",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (
                    cls.ALL_PIPELINES,
                    {"tooltip": "Which model pipeline to load"},
                ),
                "offload_mode": (
                    ["model", "sequential", "none"],
                    {
                        "default": "model",
                        "tooltip": "CPU offload strategy: 'model' (balanced VRAM/speed), 'sequential' (lowest VRAM, slowest), 'none' (fastest, highest VRAM)",
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

    def load(
        self,
        pipeline: str,
        offload_mode: str,
        hf_token: str,
        lora=None,
    ):
        # Import pipeline classes lazily
        ctor_map = {
            "sd15": lambda: __import__("src.pipelines.sd15", fromlist=["SD15Pipeline"]).SD15Pipeline,
            "sdxl": lambda: __import__("src.pipelines.sdxl", fromlist=["SDXLPipeline"]).SDXLPipeline,
            "qwen_image": lambda: __import__("src.pipelines.qwen_image", fromlist=["QwenImagePipeline"]).QwenImagePipeline,
            "qwen_image_2512": lambda: __import__("src.pipelines.qwen_image", fromlist=["QwenImage2512Pipeline"]).QwenImage2512Pipeline,
            "flux": lambda: __import__("src.pipelines.flux_dev", fromlist=["FluxDevPipeline"]).FluxDevPipeline,
            "flux_kontext": lambda: __import__("src.pipelines.flux_kontext", fromlist=["FluxKontextPipeline"]).FluxKontextPipeline,
            "flux2": lambda: __import__("src.pipelines.flux2", fromlist=["Flux2Pipeline"]).Flux2Pipeline,
            "flux2_klein_4b": lambda: __import__("src.pipelines.flux2_klein", fromlist=["Flux2Klein4BPipeline"]).Flux2Klein4BPipeline,
            "flux2_klein_9b": lambda: __import__("src.pipelines.flux2_klein", fromlist=["Flux2Klein9BPipeline"]).Flux2Klein9BPipeline,
            "flex1": lambda: __import__("src.pipelines.flex1_alpha", fromlist=["Flex1AlphaPipeline"]).Flex1AlphaPipeline,
            "flex2": lambda: __import__("src.pipelines.flex2", fromlist=["Flex2Pipeline"]).Flex2Pipeline,
            "zimage": lambda: __import__("src.pipelines.zimage", fromlist=["ZImagePipeline"]).ZImagePipeline,
            "zimage_turbo": lambda: __import__("src.pipelines.zimage_turbo", fromlist=["ZImageTurboPipeline"]).ZImageTurboPipeline,
            "zimage_deturbo": lambda: __import__("src.pipelines.zimage_deturbo", fromlist=["ZImageDeturboPipeline"]).ZImageDeturboPipeline,
            "chroma": lambda: __import__("src.pipelines.chroma", fromlist=["ChromaPipeline"]).ChromaPipeline,
            "hidream": lambda: __import__("src.pipelines.hidream", fromlist=["HiDreamPipeline"]).HiDreamPipeline,
            "hidream_e1": lambda: __import__("src.pipelines.hidream", fromlist=["HiDreamE1Pipeline"]).HiDreamE1Pipeline,
            "lumina2": lambda: __import__("src.pipelines.lumina2", fromlist=["Lumina2Pipeline"]).Lumina2Pipeline,
            "omnigen2": lambda: __import__("src.pipelines.omnigen2", fromlist=["OmniGen2Pipeline"]).OmniGen2Pipeline,
            "qwen_image_edit": lambda: __import__("src.pipelines.qwen_image", fromlist=["QwenImageEditPipeline"]).QwenImageEditPipeline,
            "qwen_image_edit_plus": lambda: __import__("src.pipelines.qwen_image", fromlist=["QwenImageEditPlus2509Pipeline"]).QwenImageEditPlus2509Pipeline,
            "qwen_image_edit_plus_2511": lambda: __import__("src.pipelines.qwen_image", fromlist=["QwenImageEditPlus2511Pipeline"]).QwenImageEditPlus2511Pipeline,
            "ltx2": lambda: __import__("src.pipelines.ltx2", fromlist=["LTX2Pipeline"]).LTX2Pipeline,
            "ltx2.3": lambda: __import__("src.pipelines.ltx2", fromlist=["LTX23Pipeline"]).LTX23Pipeline,
            "wan21_14b": lambda: __import__("src.pipelines.wan21", fromlist=["Wan21T2V14BPipeline"]).Wan21T2V14BPipeline,
            "wan21_1b": lambda: __import__("src.pipelines.wan21", fromlist=["Wan21T2V1BPipeline"]).Wan21T2V1BPipeline,
            "wan21_i2v_14b": lambda: __import__("src.pipelines.wan21", fromlist=["Wan21I2V14BPipeline"]).Wan21I2V14BPipeline,
            "wan21_i2v_14b480p": lambda: __import__("src.pipelines.wan21", fromlist=["Wan21I2V14B480PPipeline"]).Wan21I2V14B480PPipeline,
            "wan22_14b_t2v": lambda: __import__("src.pipelines.wan22_t2v", fromlist=["Wan22T2V14BPipeline"]).Wan22T2V14BPipeline,
            "wan22_14b_i2v": lambda: __import__("src.pipelines.wan22_i2v", fromlist=["Wan22I2V14BPipeline"]).Wan22I2V14BPipeline,
            "wan22_5b": lambda: __import__("src.pipelines.wan22_5b", fromlist=["Wan22TI2V5BPipeline"]).Wan22TI2V5BPipeline,
            "krea2": lambda: __import__("src.pipelines.krea2", fromlist=["Krea2Pipeline"]).Krea2Pipeline,
            "krea2_turbo": lambda: __import__("src.pipelines.krea2", fromlist=["Krea2TurboPipeline"]).Krea2TurboPipeline,
            "krea2_o_edit": lambda: __import__("src.pipelines.krea2", fromlist=["Krea2EditPipeline"]).Krea2EditPipeline,
            "krea2_o_edit_turbo": lambda: __import__("src.pipelines.krea2", fromlist=["Krea2EditTurboPipeline"]).Krea2EditTurboPipeline,
            "minimax_h3": lambda: __import__("src.pipelines.minimax_h3", fromlist=["MinimaxH3Pipeline"]).MinimaxH3Pipeline,
        }

        if pipeline not in ctor_map:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        pipeline_ctor = ctor_map[pipeline]()

        lora_paths = []
        lora_scale = 1.0

        if lora is not None:
            import folder_paths

            override = (lora.get("lora_path_or_url") or "").strip()
            name = (lora.get("lora_name") or "").strip()
            if override:
                # Pass-through: allow URL or absolute/local filesystem path
                lora_paths = [override]
            elif name:
                # Dropdown selection: resolve via ComfyUI's loras folder
                lora_paths = [folder_paths.get_full_path_or_raise("loras", name)]
            lora_scale = float(lora.get("lora_scale", 1.0))

        # Empty token -> None
        token = hf_token.strip() or None

        pipe = get_or_load_pipeline(
            model_id=pipeline,
            pipeline_ctor=pipeline_ctor,
            lora_paths=lora_paths,
            lora_scale=lora_scale,
            offload_mode=offload_mode,
            hf_token=token,
        )

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

        meta = _latent_metadata_from_pipe(pipe, samples, width=width, height=height)

        # Return with marker indicating this is an "empty" latent for txt2img
        return ({
            "samples": samples,
            "aitk_empty": True,
            **meta,
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
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**31 - 1}),
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

        _validate_latent_compat(pipe, latent)

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

        # Comfy-native progress + interrupt: wrap the run in a Comfy observer context.
        # This is a no-op outside ComfyUI.
        from src.pipelines.comfy_callbacks import comfy_pipeline_observer

        with comfy_pipeline_observer(int(steps)):
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
        meta = _latent_metadata_from_pipe(pipe, out_latents, width=width, height=height)
        return ({
            "samples": out_latents,
            **meta,
        },)


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

        _validate_latent_compat(pipe, latent)

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
        width = int(image.shape[2]) if hasattr(image, "shape") else None
        height = int(image.shape[1]) if hasattr(image, "shape") else None

        # Use batch encode if available (more efficient for batches)
        if hasattr(pipe, "encode_images_to_latents"):
            # Convert ComfyUI [B,H,W,C] in [0,1] to model [B,C,H,W] in [-1,1]
            img_tensor = _comfy_batch_to_tensor(image)
            latents = pipe.encode_images_to_latents(img_tensor)
            meta = _latent_metadata_from_pipe(pipe, latents, width=width, height=height)
            return ({"samples": latents, **meta},)

        # Fallback: encode each image separately
        pil_list = _comfy_batch_to_pil_list(image)
        latents = []
        for pil in pil_list:
            lat = pipe.encode_image_to_latent(pil)
            latents.append(lat)

        out = torch.cat(latents, dim=0)
        meta = _latent_metadata_from_pipe(pipe, out, width=width, height=height)
        return ({"samples": out, **meta},)


class RCAITKGenerate:
    """Generate images/video using a loaded pipeline.
    
    This is a unified generation node that works with any ai-toolkit pipeline.
    For models that support latent workflow (SD15, SDXL, Qwen Image), you can
    alternatively use the EmptyLatent -> Sampler -> Decode pattern for more control.
    
    Video models return IMAGE batches where the batch dimension is frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "a beautiful landscape, high quality, photorealistic"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2**31 - 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "control_image": ("IMAGE",),
                "control_image_2": ("IMAGE",),
                "control_image_3": ("IMAGE",),
                "num_frames": ("INT", {"default": 41, "min": 1, "max": 201, "tooltip": "For video models only"}),
                "fps": ("INT", {"default": _FPS_USE_MODEL_DEFAULT, "min": 0, "max": 120, "tooltip": "For video models only. 0 uses the model's own fps (e.g. MiniMax-H3 is fixed at 24); any other value is passed through, and a model that cannot honour it will say so."}),
                "kv_cache": ("BOOLEAN", {"default": True, "tooltip": "Krea 2 edit only. Must match the training config's model_kwargs.kv_cache -- it changes the reference attention mask, so a mismatch silently produces wrong output. Set false for adapters trained before 2026-07-16. Ignored by other models."}),
                "match_target_res": ("BOOLEAN", {"default": True, "tooltip": "Krea 2 edit only. Must match the training config's model_kwargs.match_target_res. Ignored by other models."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = WORKFLOW_CATEGORY

    def generate(
        self,
        pipe,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        negative_prompt: str = "",
        control_image=None,
        control_image_2=None,
        control_image_3=None,
        num_frames: int = 41,
        fps: int = 16,
        kv_cache: bool = True,
        match_target_res: bool = True,
    ):
        # Convert control images from ComfyUI format to PIL
        ctrl_img = None
        ctrl_imgs = None
        
        if control_image is not None:
            ctrl_img = comfy_to_pil_image(control_image)
        
        extras = []
        if control_image_2 is not None:
            extras.append(comfy_to_pil_image(control_image_2))
        if control_image_3 is not None:
            extras.append(comfy_to_pil_image(control_image_3))
        
        if extras:
            base = [ctrl_img] if ctrl_img is not None else []
            ctrl_imgs = base + extras

        # Check if this is a video model. NOTE the field is is_video_model --
        # PipelineConfig has never had an `is_video` attribute, so the old
        # getattr(..., "is_video", False) silently evaluated False for every
        # video model and dropped num_frames/fps from the generate() call.
        is_video = getattr(getattr(pipe, "CONFIG", None), "is_video_model", False)

        effective_fps = _resolve_fps(
            fps, getattr(getattr(pipe, "CONFIG", None), "default_fps", None)
        )

        # Model-specific options that must match how the LoRA was trained.
        # BasePipeline.apply_model_options() is a no-op for models that declare
        # none, so this is safe to call unconditionally.
        try:
            pipe.apply_model_options(
                kv_cache=bool(kv_cache), match_target_res=bool(match_target_res)
            )
        except Exception as e:
            logger.warning(f"Failed to apply model options: {e}")

        # Comfy-native progress + interrupt
        from src.pipelines.comfy_callbacks import comfy_pipeline_observer

        with comfy_pipeline_observer(int(steps)):
            result = pipe.generate(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                seed=int(seed),
                control_image=ctrl_img,
                control_images=ctrl_imgs,
                num_frames=int(num_frames) if is_video else None,
                fps=effective_fps if is_video else None,
            )

        # Handle result
        if "image" in result:
            return (pil_to_comfy_image(result["image"]),)

        frames = result.get("frames")
        if frames:
            return (pil_frames_to_comfy_images(frames),)

        video_tensor = result.get("video_tensor")
        if video_tensor is not None:
            return (video_tensor_to_comfy_images(video_tensor),)

        raise ValueError(f"Unexpected pipeline result keys: {list(result.keys())}")


# ===== LTX-2.3 Composable Upscale Workflow (3-node pattern) =====
#
# Community-standard 3-stage workflow for LTX-2.3 spatial upscale:
#   GenerateLatent → LatentUpscale → Denoise
#
# Wire type: LTX23_LATENT (dict with video_latents, audio_latents, metadata)


class RCLTX23GenerateLatent:
    """Stage 1: Generate LTX-2.3 video at given resolution, output raw latent.

    This is the first node in the 3-stage upscale workflow.
    Connect the output to RCLTX23LatentUpscale for 2x spatial upscale.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 0x7FFFFFFF}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "control_image": ("IMAGE",),
                "num_frames": ("INT", {"default": 41, "min": 1, "max": 201}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            },
        }

    RETURN_TYPES = ("LTX23_LATENT",)
    RETURN_NAMES = ("ltx_latent",)
    FUNCTION = "generate"
    CATEGORY = WORKFLOW_CATEGORY

    def generate(
        self,
        pipe,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        negative_prompt: str = "",
        control_image=None,
        num_frames: int = 41,
        fps: int = 24,
    ):
        if not hasattr(pipe, "generate_latent"):
            raise ValueError(
                "RCLTX23GenerateLatent requires an LTX-2.3 pipeline. "
                "Use RCAITKLoadPipeline with pipeline='ltx2.3'."
            )

        import torch as _torch

        if seed < 0:
            seed = _torch.randint(0, 2147483647, (1,)).item()
        generator = _torch.Generator(device="cpu").manual_seed(int(seed))

        ctrl_img = None
        if control_image is not None:
            ctrl_img = comfy_to_pil_image(control_image)

        from src.pipelines.comfy_callbacks import comfy_pipeline_observer

        with comfy_pipeline_observer(int(steps)):
            result = pipe.generate_latent(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                generator=generator,
                control_image=ctrl_img,
                num_frames=int(num_frames),
                fps=int(fps),
            )

        return (result,)


class RCLTX23LatentUpscale:
    """Stage 2: 2x spatial upsample of LTX-2.3 video latent.

    Takes LTX23_LATENT from GenerateLatent, outputs upscaled LTX23_LATENT.
    Connect the output to RCLTX23Denoise.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "ltx_latent": ("LTX23_LATENT",),
            },
        }

    RETURN_TYPES = ("LTX23_LATENT",)
    RETURN_NAMES = ("ltx_latent",)
    FUNCTION = "upscale"
    CATEGORY = WORKFLOW_CATEGORY

    def upscale(self, pipe, ltx_latent):
        if not hasattr(pipe, "latent_upscale"):
            raise ValueError(
                "RCLTX23LatentUpscale requires an LTX-2.3 pipeline. "
                "Use RCAITKLoadPipeline with pipeline='ltx2.3'."
            )

        video_latents = ltx_latent["video_latents"]
        upscaled = pipe.latent_upscale(video_latents)

        # Return updated dict with upscaled latents and doubled dimensions
        return ({
            **ltx_latent,
            "video_latents": upscaled,
            "width": ltx_latent["width"] * 2,
            "height": ltx_latent["height"] * 2,
        },)


class RCLTX23Denoise:
    """Stage 3: Denoise upscaled LTX-2.3 latent with distilled LoRA.

    Takes upscaled LTX23_LATENT from LatentUpscale, outputs decoded video frames.
    This is the terminal node in the 3-stage workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("AITK_PIPELINE",),
                "ltx_latent": ("LTX23_LATENT",),
            },
            "optional": {
                "steps": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 50,
                        "tooltip": "Denoise steps (default 3 for distilled LoRA)",
                    },
                ),
                "noise_scale": (
                    "FLOAT",
                    {
                        "default": 0.909375,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Initial sigma / noise scale",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "denoise"
    CATEGORY = WORKFLOW_CATEGORY

    def denoise(
        self,
        pipe,
        ltx_latent,
        steps: int = 3,
        noise_scale: float = 0.909375,
    ):
        if not hasattr(pipe, "denoise_latent"):
            raise ValueError(
                "RCLTX23Denoise requires an LTX-2.3 pipeline. "
                "Use RCAITKLoadPipeline with pipeline='ltx2.3'."
            )

        video_latents = ltx_latent["video_latents"]
        audio_latents = ltx_latent["audio_latents"]
        prompt = ltx_latent["prompt"]
        negative_prompt = ltx_latent.get("negative_prompt", "")
        width = ltx_latent["width"]
        height = ltx_latent["height"]
        num_frames = ltx_latent.get("num_frames", 41)
        fps = ltx_latent.get("fps", 24)

        # Build sigma schedule: evenly spaced from noise_scale to ~0 for given steps
        sigma_values = None
        if steps == 3 and abs(noise_scale - 0.909375) < 1e-6:
            # Use official defaults
            sigma_values = None
        else:
            # Generate evenly spaced sigmas from noise_scale down
            import numpy as _np
            sigma_values = list(_np.linspace(noise_scale, noise_scale / steps, steps))

        from src.pipelines.comfy_callbacks import comfy_pipeline_observer

        with comfy_pipeline_observer(int(steps)):
            result = pipe.denoise_latent(
                video_latents=video_latents,
                audio_latents=audio_latents,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_frames=int(num_frames),
                fps=int(fps),
                num_inference_steps=int(steps),
                sigma_values=sigma_values,
            )

        # Convert uint8 [T,H,W,C] tensor to ComfyUI float32 [B,H,W,C] in [0,1]
        video_tensor = result["video_tensor"]
        frames = video_tensor.float() / 255.0

        return (frames,)
