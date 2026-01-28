"""
Wan 2.2 TI2V 5B pipeline implementation.

Supports both T2V (text-to-video) and I2V (image-to-video) modes.
- T2V: No control_image provided
- I2V: control_image provided -> uses first frame conditioning

Note: This is a Tier 2 model that requires ai-toolkit.
Set AI_TOOLKIT_PATH environment variable to enable.
"""

import os
import logging
from typing import Dict, Any, Optional
import sys

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .base import BasePipeline, PipelineConfig
from ..schemas.models import ModelType
from ..config import settings

logger = logging.getLogger(__name__)

# align withe aitk
SCHEDULER_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 5.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}


class Wan22TI2V5BPipeline(BasePipeline):
    """
    Wan 2.2 TI2V 5B pipeline.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.WAN22_5B,
        base_model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        resolution_divisor=32,
        default_steps=25,
        default_guidance_scale=4.0,
        is_video_model=True,
        default_num_frames=41,
        default_fps=16,
        enable_cpu_offload=False,  # Disabled: VAE must be on GPU for I2V conditioning
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder_list = None
        self.tokenizer_list = None
        self.transformer = None

    def _load_pipeline(self):
        """
        Load Wan 2.2 T2V 5B pipeline.

        Memory management: Only flush after loading transformer (largest component),
        aligned with ai-toolkit's approach in wan21.py.
        """
        # Add ai-toolkit to path (configurable via AI_TOOLKIT_PATH env var)
        ai_toolkit_path = settings.ai_toolkit_path
        if os.path.exists(ai_toolkit_path) and ai_toolkit_path not in sys.path:
            sys.path.insert(0, ai_toolkit_path)

        from diffusers import UniPCMultistepScheduler
        from diffusers import WanTransformer3DModel, AutoencoderKLWan
        from transformers import AutoTokenizer, UMT5EncoderModel
        from extensions_built_in.diffusion_models.wan22.wan22_pipeline import Wan22Pipeline
        from toolkit.basic import flush

        dtype = torch.bfloat16

        # Load VAE
        vae = AutoencoderKLWan.from_pretrained(
            self.CONFIG.base_model,
            subfolder="vae",
            torch_dtype=dtype,
        )
        logger.info(f"Loaded VAE from {self.CONFIG.base_model}")

        # Load Transformer (largest component)
        transformer = WanTransformer3DModel.from_pretrained(
            self.CONFIG.base_model,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        logger.info(f"Loaded Transformer from {self.CONFIG.base_model}")

        # Flush after loading large transformer (aligned with ai-toolkit)
        flush()

        # Load Text Encoder
        text_encoder = UMT5EncoderModel.from_pretrained(
            self.CONFIG.base_model,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        logger.info(f"Loaded Text Encoder from {self.CONFIG.base_model}")

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.CONFIG.base_model,
            subfolder="tokenizer",
        )
        logger.info(f"Loaded Tokenizer from {self.CONFIG.base_model}")

        # Create scheduler
        scheduler = UniPCMultistepScheduler(**SCHEDULER_CONFIG)
        logger.info(f"Created scheduler with flow_shift={SCHEDULER_CONFIG['flow_shift']}")

        # Create pipeline
        self.pipe = Wan22Pipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            transformer_2=transformer,  # 5B: use the same transformer
            scheduler=scheduler,
            expand_timesteps=True,  # 5B special
            device=torch.device("cuda"),
            aggressive_offload=False,
        )
        logger.info(f"Wan22Pipeline created successfully")

        # Move all components to GPU (required for I2V conditioning with VAE)
        # Aligned with wan22_i2v.py approach
        logger.info(f"Moving components to {self.device}...")
        self.pipe.vae.to(self.device)
        self.pipe.text_encoder.to(self.device)
        self.pipe.transformer.to(self.device)

    def _prepare_i2v_conditioning(
        self,
        control_image: Image.Image,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
    ) -> tuple:
        """
        Prepare first frame conditioning for I2V mode.

        Returns:
            (latents, noise_mask) tuple for I2V inference
        """
        from toolkit.models.wan21.wan_utils import add_first_frame_conditioning_v22

        device = self.device
        dtype = self.dtype

        # Ensure VAE is on device for encoding (critical for I2V)
        self.pipe.vae.to(device)

        # Resize control image to match output dimensions
        control_image = control_image.resize((width, height), Image.LANCZOS)

        # Prepare initial latents (aligned with AITK)
        num_channels_latents = self.pipe.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            1,  # batch_size
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            torch.device(device),
            generator,
            None,
        ).to(dtype)

        # Convert control image to tensor [-1, 1]
        first_frame_n1p1 = TF.to_tensor(control_image).unsqueeze(0).to(device, dtype=dtype) * 2.0 - 1.0

        # Apply first frame conditioning (aligned with AITK)
        conditioned_latents, noise_mask = add_first_frame_conditioning_v22(
            latent_model_input=latents,
            first_frame=first_frame_n1p1,
            vae=self.pipe.vae,
        )

        return conditioned_latents, noise_mask

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        control_image: Optional[Image.Image] = None,
        control_images: Optional[list] = None,
        num_frames: int = 41,
        fps: int = 16,
    ) -> Dict[str, Any]:
        """
        Run Wan 2.2 5B inference.

        Supports both T2V and I2V modes:
        - T2V: No control_image -> standard text-to-video
        - I2V: control_image provided -> first frame conditioning
        """
        # Ensure num_frames is divisible by 4 + 1 (aligned with ai-toolkit)
        num_frames = ((num_frames - 1) // 4) * 4 + 1

        latents = None
        noise_mask = None

        if control_image is not None:
            # I2V mode: prepare first frame conditioning
            logger.info("Running I2V inference (control_image provided)")
            latents, noise_mask = self._prepare_i2v_conditioning(
                control_image=control_image,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
            )
        else:
            # T2V mode: standard text-to-video
            logger.info("Running T2V inference")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
            noise_mask=noise_mask,
            output_type="pil",
        )

        frames = result.frames[0] if hasattr(result, "frames") else result.images

        return {"frames": frames, "fps": fps}
