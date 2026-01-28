"""
Image utility functions for loading and encoding images.
"""

import base64
import io
import os
from typing import Optional, List
import logging

import torch
import numpy as np
from PIL import Image
import requests

logger = logging.getLogger(__name__)


def load_image_from_source(
    base64_data: Optional[str] = None,
    url: Optional[str] = None,
    local_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Load an image from various sources.

    Args:
        base64_data: Base64 encoded image data
        url: Image URL
        local_path: Local file path

    Returns:
        PIL Image or None if no source provided

    Raises:
        ValueError: If image cannot be loaded
    """
    if base64_data:
        try:
            # Handle data URL format (data:image/png;base64,...)
            if base64_data.startswith("data:"):
                base64_data = base64_data.split(",", 1)[1]

            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")

    if url:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")

    if local_path:
        if not os.path.exists(local_path):
            raise ValueError(f"Image file not found: {local_path}")
        try:
            image = Image.open(local_path)
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from file: {e}")

    return None


def encode_image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """
    Encode a PIL Image to base64 string.

    Args:
        image: PIL Image to encode
        format: Output format (JPEG, PNG, WEBP)
        quality: JPEG/WEBP quality (1-100)

    Returns:
        Base64 encoded image string
    """
    buffer = io.BytesIO()

    # Ensure RGB for JPEG
    if format.upper() == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    save_kwargs = {"format": format.upper()}
    if format.upper() in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality

    image.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_image(image: Image.Image, output_path: str, format: str = "JPEG", quality: int = 95):
    """
    Save a PIL Image to file.

    Args:
        image: PIL Image to save
        output_path: Output file path
        format: Output format (JPEG, PNG, WEBP)
        quality: JPEG/WEBP quality (1-100)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure RGB for JPEG
    if format.upper() == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    save_kwargs = {}
    if format.upper() in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality

    image.save(output_path, format=format.upper(), **save_kwargs)
    logger.info(f"Saved image to {output_path}")


def save_video_frames(frames: List[Image.Image], output_path: str, fps: int = 16, format: str = "WEBP") -> str:
    """
    Save video frames as animated image.

    Args:
        frames: List of PIL Images
        output_path: Output file path
        fps: Frames per second
        format: Output format (WEBP, GIF)

    Returns:
        Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    duration = int(1000 / fps)  # Duration per frame in milliseconds

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        format=format.upper(),
    )

    logger.info(f"Saved video to {output_path}")
    return output_path


def encode_video_to_base64(frames: List[Image.Image], fps: int = 16, format: str = "WEBP") -> str:
    """
    Encode video frames to base64 string.

    Args:
        frames: List of PIL Images
        fps: Frames per second
        format: Output format (WEBP, GIF)

    Returns:
        Base64 encoded video string
    """
    buffer = io.BytesIO()
    duration = int(1000 / fps)

    frames[0].save(
        buffer,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        format=format.upper(),
    )

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_video_with_audio(
    output_path: str,
    fps: int = 24,
    audio: Optional[torch.Tensor] = None,
    audio_sample_rate: int = 24000,
    video_tensor: Optional[torch.Tensor] = None,
    frames: Optional[List[Image.Image]] = None,
) -> str:
    """
    Save video with audio as MP4.

    Uses diffusers.pipelines.ltx2.export_utils.encode_video for MP4 encoding.
    Aligned with ai-toolkit's training sample handling.

    Args:
        output_path: Output file path (should end with .mp4)
        fps: Frames per second
        audio: Audio waveform tensor [channels, samples] (optional)
        audio_sample_rate: Audio sample rate in Hz (default 24000)
        video_tensor: Video tensor [T, C, H, W] uint8 (preferred, from LTX-2)
        frames: List of PIL Images (fallback, converted to tensor)

    Returns:
        Output file path
    """
    from diffusers.pipelines.ltx2.export_utils import encode_video

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use video_tensor if provided, otherwise convert from PIL frames
    if video_tensor is not None:
        # Already in [T, C, H, W] format from LTX-2 pipeline
        video = video_tensor
    elif frames is not None:
        # Convert PIL frames to video tensor [T, C, H, W] uint8
        video_np = np.stack([np.array(frame) for frame in frames], axis=0)  # [T, H, W, C]
        video_np = video_np.astype(np.uint8)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # [T, C, H, W]
    else:
        raise ValueError("Either video_tensor or frames must be provided")

    # Prepare audio (move to CPU and float if provided)
    audio_cpu = None
    if audio is not None:
        audio_cpu = audio.float().cpu()

    # Encode video with audio
    encode_video(
        video=video,
        fps=fps,
        audio=audio_cpu,
        audio_sample_rate=audio_sample_rate,
        output_path=output_path,
    )

    logger.info(f"Saved video with audio to {output_path}")
    return output_path
