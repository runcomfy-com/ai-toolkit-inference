"""
Model type definitions.

Model types are named to match inference_schema JSON file names (without .json extension).
Each pipeline class defines its own configuration via class attributes.
"""

from enum import Enum


class ModelType(str, Enum):
    """
    Supported model types.

    """

    # FLUX family
    FLUX = "flux"
    FLUX_KONTEXT = "flux_kontext"
    FLUX2 = "flux2"
    FLUX2_DIFFUSERS = "flux2_diffusers"
    FLUX2_KLEIN_4B = "flux2_klein_4b"
    FLUX2_KLEIN_9B = "flux2_klein_9b"

    # Flex family
    FLEX1 = "flex1"
    FLEX2 = "flex2"

    # Stable Diffusion family
    SD15 = "sd15"
    SDXL = "sdxl"

    # Qwen family
    QWEN_IMAGE = "qwen_image"
    QWEN_IMAGE_2512 = "qwen_image_2512"
    QWEN_IMAGE_EDIT = "qwen_image_edit"
    QWEN_IMAGE_EDIT_PLUS_2509 = "qwen_image_edit_plus"
    QWEN_IMAGE_EDIT_PLUS_2511 = "qwen_image_edit_plus_2511"

    # Z-Image family
    ZIMAGE_TURBO = "zimage_turbo"
    ZIMAGE_DETURBO = "zimage_deturbo"

    # Wan 2.1 family
    WAN21_14B = "wan21_14b"
    WAN21_1B = "wan21_1b"
    WAN21_I2V_14B = "wan21_i2v_14b"
    WAN21_I2V_14B_480P = "wan21_i2v_14b480p"

    # Wan 2.2 family
    WAN22_14B_T2V = "wan22_14b_t2v"
    WAN22_14B_I2V = "wan22_14b_i2v"
    WAN22_5B = "wan22_5b"

    # Chroma
    CHROMA = "chroma"  # chroma.json

    # HiDream
    HIDREAM = "hidream"  # hidream.json
    HIDREAM_E1 = "hidream_e1"  # hidream_e1.json

    # Lumina
    LUMINA2 = "lumina2"  # lumina2.json

    # OmniGen
    OMNIGEN2 = "omnigen2"  # omnigen2.json

    # LTX-2
    LTX2 = "ltx2"  # ltx2.json (supports both T2V and I2V via control_image parameter)


def get_supported_models() -> list[str]:
    """Get list of all supported model type values."""
    return [m.value for m in ModelType]
