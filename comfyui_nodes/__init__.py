"""ComfyUI nodes for ai-toolkit-inference pipelines."""
from .rc_models import (
    RCZimage,
    RCZimageTurbo,
    RCZimageDeturbo,
    RCFluxDev,
    RCFluxKontext,
    RCFlux2,
    RCFlux2Klein4B,
    RCFlux2Klein9B,
    RCFlex1,
    RCFlex2,
    RCSD15,
    RCSDXL,
    RCQwenImage,
    RCQwenImage2512,
    RCQwenImageEdit,
    RCQwenImageEditPlus,
    RCQwenImageEditPlus2511,
    RCChroma,
    RCHiDream,
    RCHiDreamE1,
    RCLumina2,
    RCOmniGen2,
    RCLTX2,
    RCWan21T2V14B,
    RCWan21T2V1B,
    RCWan21I2V14B,
    RCWan21I2V14B480P,
    RCWan22T2V14B,
    RCWan22I2V14B,
    RCWan22TI2V5B,
)

NODE_CLASS_MAPPINGS = {
    # Z-Image
    "RCZimage": RCZimage,
    "RCZimageTurbo": RCZimageTurbo,
    "RCZimageDeturbo": RCZimageDeturbo,

    # FLUX
    "RCFluxDev": RCFluxDev,
    "RCFluxKontext": RCFluxKontext,
    "RCFlux2": RCFlux2,
    "RCFlux2Klein4B": RCFlux2Klein4B,
    "RCFlux2Klein9B": RCFlux2Klein9B,

    # Flex
    "RCFlex1": RCFlex1,
    "RCFlex2": RCFlex2,

    # Stable Diffusion
    "RCSD15": RCSD15,
    "RCSDXL": RCSDXL,

    # Qwen
    "RCQwenImage": RCQwenImage,
    "RCQwenImage2512": RCQwenImage2512,
    "RCQwenImageEdit": RCQwenImageEdit,
    "RCQwenImageEditPlus": RCQwenImageEditPlus,
    "RCQwenImageEditPlus2511": RCQwenImageEditPlus2511,

    # Other image models
    "RCChroma": RCChroma,
    "RCHiDream": RCHiDream,
    "RCHiDreamE1": RCHiDreamE1,
    "RCLumina2": RCLumina2,
    "RCOmniGen2": RCOmniGen2,

    # Video models
    "RCLTX2": RCLTX2,
    "RCWan21T2V14B": RCWan21T2V14B,
    "RCWan21T2V1B": RCWan21T2V1B,
    "RCWan21I2V14B": RCWan21I2V14B,
    "RCWan21I2V14B480P": RCWan21I2V14B480P,
    "RCWan22T2V14B": RCWan22T2V14B,
    "RCWan22I2V14B": RCWan22I2V14B,
    "RCWan22TI2V5B": RCWan22TI2V5B,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RCZimage": "RC Z-Image",
    "RCZimageTurbo": "RC Z-Image Turbo",
    "RCZimageDeturbo": "RC Z-Image De-Turbo",

    "RCFluxDev": "RC FLUX.1-dev",
    "RCFluxKontext": "RC FLUX Kontext",
    "RCFlux2": "RC FLUX.2",
    "RCFlux2Klein4B": "RC FLUX.2-klein 4B",
    "RCFlux2Klein9B": "RC FLUX.2-klein 9B",

    "RCFlex1": "RC Flex.1-alpha",
    "RCFlex2": "RC Flex.2",

    "RCSD15": "RC SD 1.5",
    "RCSDXL": "RC SDXL",

    "RCQwenImage": "RC Qwen Image",
    "RCQwenImage2512": "RC Qwen Image 2512",
    "RCQwenImageEdit": "RC Qwen Image Edit",
    "RCQwenImageEditPlus": "RC Qwen Image Edit Plus",
    "RCQwenImageEditPlus2511": "RC Qwen Image Edit Plus 2511",

    "RCChroma": "RC Chroma",
    "RCHiDream": "RC HiDream",
    "RCHiDreamE1": "RC HiDream E1",
    "RCLumina2": "RC Lumina2",
    "RCOmniGen2": "RC OmniGen2",

    "RCLTX2": "RC LTX-2",
    "RCWan21T2V14B": "RC Wan 2.1 T2V 14B",
    "RCWan21T2V1B": "RC Wan 2.1 T2V 1B",
    "RCWan21I2V14B": "RC Wan 2.1 I2V 14B",
    "RCWan21I2V14B480P": "RC Wan 2.1 I2V 14B 480p",
    "RCWan22T2V14B": "RC Wan 2.2 T2V 14B",
    "RCWan22I2V14B": "RC Wan 2.2 I2V 14B",
    "RCWan22TI2V5B": "RC Wan 2.2 TI2V 5B",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
