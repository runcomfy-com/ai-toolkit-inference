"""
MiniMax-H3 pipeline (fl2va partition): video + joint stereo audio.

MiniMax-H3 jointly denoises video and audio rows in one packed sequence:
  - denoiser:      MiniMaxH3Transformer, 50 blocks, shipped pre-quantized as
                   int8 ConvRot (~21 GB)
  - text encoder:  Qwen3-VL-32B truncated to 50 decoder layers, shipped
                   pre-quantized as NVFP4 AWQ (~16 GB)
  - video VAE:     f16 spatial, fp16 weights
  - audio VAE:     32 kHz stereo, fp32 weights

There is no diffusers pipeline for MiniMax-H3, and ai-toolkit's MinimaxH3Model
is a BaseModel (not a DiffusionPipeline), so this wrapper builds the four
components by hand and then hands them to ai-toolkit's OWN sampler,
extensions_built_in/diffusion_models/minimax_h3/src/pipeline.py, which is the
released sampling implementation and is exactly what the trainer calls to render
its preview samples (minimax_h3.py:857-909). Importing it rather than
reimplementing the Euler loop is the entire parity story: the dual video/audio
sigma schedules, the packed-sequence layout, the keyframe conditioning rows and
the decode path cannot drift from training, because they are the same code.

Those leaf modules import only torch / numpy / PIL and one diffusers helper --
no toolkit.* -- so they load standalone without dragging in ai-toolkit's
21-architecture __init__.

LORA IS ATTACHED LIVE, NOT MERGED. See _H3LoraLinear: merging into an
int8-ConvRot base is dequantize -> add -> requantize, which resamples every row
scale (ai-toolkit toolkit/network_mixins.py:381-384, :407-410 document ~0.1%
output drift per merge cycle per layer). Training never merges either --
LoRAModule.forward is org_forward(x) + lora_output * scale
(network_mixins.py:304-315). Merging here would inject a requantization round
the training preview never performed, on every LoRA-touched layer across 50
blocks, and the resulting drift is indistinguishable from "quantization is
lossy" when eyeballing a parity test.

IMPORT SAFETY: every heavy import lives inside a method. src/api/v1/inference.py
imports PIPELINE_REGISTRY at module scope and src/pipelines/__init__.py builds it
with no try/except, so a top-level import failure here would break FastAPI
startup for every model in the catalog.
"""

import gc
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .base import BasePipeline, LoraMergeMethod, PipelineConfig
from ..config import settings
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)

# ai-toolkit extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py:93-102
COMFY_REPO = "Comfy-Org/MiniMax-H3"
COMFY_FILES = {
    "dit_fl2va": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
}
# tokenizer / processor / text-encoder config come from the original repo
ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"

# only hidden_states[50] is consumed, so the decoder stack is truncated to 50
# layers and the final norm neutralized (minimax_h3.py:355-360)
TEXT_ENCODER_LAYER = 50

# packing.py constants, restated here only for the CONFIG defaults
AUDIO_SAMPLE_RATE = 32000
FPS = 24


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_AITK_MODULES: Optional[Dict[str, Any]] = None

_H3_REL_PATH = os.path.join(
    "extensions_built_in", "diffusion_models", "minimax_h3", "src"
)


def _find_h3_src_dir() -> str:
    """Locate ai-toolkit's minimax_h3/src directory.

    Same resolution chain as krea2.py:89-136 -- settings.ai_toolkit_path first,
    then wherever `extensions_built_in` already resolves on sys.path (the
    production image exposes ai-toolkit through PYTHONPATH, not
    AI_TOOLKIT_PATH). find_spec() locates the package without executing its
    __init__.
    """
    import importlib.util

    tried: List[str] = []

    def _try(candidate: str) -> bool:
        if candidate in tried:
            return False
        tried.append(candidate)
        return os.path.isdir(candidate)

    configured = settings.ai_toolkit_path
    if configured:
        candidate = os.path.join(configured, _H3_REL_PATH)
        if _try(candidate):
            return candidate

    try:
        spec = importlib.util.find_spec("extensions_built_in")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidate = os.path.join(
                location, "diffusion_models", "minimax_h3", "src"
            )
            if _try(candidate):
                return candidate

    raise ImportError(
        "MiniMax-H3 requires an ai-toolkit checkout containing "
        "extensions_built_in/diffusion_models/minimax_h3/ (ai-toolkit >= 0.12.2). "
        "Tried:\n  " + "\n  ".join(tried) + "\n"
        "Set AI_TOOLKIT_PATH, put ai-toolkit on PYTHONPATH, or run install.py."
    )


def _load_aitk_h3_modules() -> Dict[str, Any]:
    """Load ai-toolkit's minimax_h3 leaf modules by file path.

    Deliberately does NOT go through `import extensions_built_in.diffusion_models
    .minimax_h3...`: that package's __init__ eagerly imports every model
    architecture, dragging in toolkit.* and a large chunk of import cost that
    H3 does not need. These six modules import only torch, numpy, PIL and
    diffusers.utils.torch_utils.randn_tensor.

    This still executes upstream's own source -- nothing is vendored or
    reimplemented -- so the sampling math cannot drift from the trainer.
    """
    global _AITK_MODULES
    if _AITK_MODULES is not None:
        return _AITK_MODULES

    import importlib.util
    from types import ModuleType

    src_dir = _find_h3_src_dir()

    # A synthetic parent package so `from .packing import ...` inside
    # pipeline.py resolves without touching the real extensions_built_in.
    pkg_name = "_aitk_minimax_h3_src"
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [src_dir]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    modules: Dict[str, Any] = {}
    # dependency order: packing is imported relatively by transformer and
    # pipeline; pipeline imports all of the others.
    for name in (
        "packing",
        "transformer",
        "vae",
        "audio_vae",
        "text_encoder",
        "pipeline",
    ):
        full_name = f"{pkg_name}.{name}"
        if full_name in sys.modules:
            modules[name] = sys.modules[full_name]
            continue
        path = os.path.join(src_dir, f"{name}.py")
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load MiniMax-H3 module from {path!r}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(full_name, None)
            raise
        modules[name] = module

    _AITK_MODULES = modules
    return modules


class _H3LoraLinear(torch.nn.Module):
    """Live additive LoRA over a (possibly quantized) linear.

    Mirrors ai-toolkit's LoRAModule.forward (toolkit/network_mixins.py:304-315):
    ``org_forward(x) + lora_up(lora_down(x)) * scale``.

    Deliberately NOT a weight merge. On the int8-ConvRot base a merge is
    dequantize -> add -> requantize, which resamples every row scale
    (network_mixins.py:381-384) with ~0.1% output drift per cycle per layer
    (:407-410). The trainer's preview never merges, so merging here would make
    inference structurally different from the thing we are trying to match.
    """

    def __init__(self, base: torch.nn.Module, down: torch.Tensor, up: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        # kept as plain buffers: these are frozen at inference
        self.register_buffer("down", down, persistent=False)
        self.register_buffer("up", up, persistent=False)
        self.scale = float(scale)

    def forward(self, x, *args, **kwargs):
        out = self.base(x, *args, **kwargs)
        if self.scale == 0.0:
            return out
        h = torch.nn.functional.linear(x.to(self.down.dtype), self.down)
        delta = torch.nn.functional.linear(h, self.up)
        return out + delta.to(out.dtype) * self.scale


class _ModelShim:
    """The ~20-line stand-in for ai-toolkit's BaseModel.

    src/pipeline.py touches exactly these members, nothing else:
      .device_torch, .torch_dtype, .transformer,
      .encode_keyframe_latents(), .decode_latents(), .decode_audio_latents()
    """

    def __init__(self, transformer, video_vae, audio_vae, device, dtype, packing):
        self.transformer = transformer
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.device_torch = device
        self.torch_dtype = dtype
        self._packing = packing

    def encode_keyframe_latents(self, frames: torch.Tensor) -> torch.Tensor:
        """(B,3,1,H,W) in [-1,1] -> normalized latents (B,24,1,h,w).

        Ported from minimax_h3.py:617-631. The seed-42 posterior sample and the
        fp16 rounding are the released conditioning recipe and are load-bearing:
        the seed is deliberately independent of the request seed.
        """
        generator = torch.Generator(device="cpu").manual_seed(
            self._packing.KEYFRAME_ENCODE_SEED
        )
        latents = self.video_vae.encode(
            frames.to(self.video_vae.device, self.video_vae.dtype),
            sample=True,
            generator=generator,
            fp16_round=True,
        )
        return latents.float()

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        video = self.video_vae.decode(
            latents.to(self.video_vae.device, self.video_vae.dtype)
        )
        if device is not None:
            video = video.to(device, dtype=dtype)
        return video

    def decode_audio_latents(self, latents: torch.Tensor):
        """(B,32,T) normalized -> waveform (B,1,T*800) at 32 kHz."""
        return self.audio_vae.decode(
            latents.to(self.audio_vae.device, torch.float32)
        )


class MinimaxH3Pipeline(BasePipeline):
    """MiniMax-H3 fl2va: text-to-video and first-frame image-to-video, with
    jointly generated stereo audio."""

    CONFIG = PipelineConfig(
        model_type=ModelType.MINIMAX_H3,
        base_model=COMFY_REPO,
        # 16x VAE spatial compression * 2x2 transformer patch (minimax_h3.py:184-186)
        resolution_divisor=32,
        default_steps=28,  # ai-toolkit ui/src/app/jobs/new/options.tsx
        # guidance-distilled: the sampler has no CFG path at all and ignores this
        default_guidance_scale=1.0,
        requires_control_image=False,  # present -> first-frame i2v, absent -> t2v
        supports_negative_prompt=False,  # no unconditional branch exists
        is_video_model=True,
        default_num_frames=107,
        default_fps=FPS,
        lora_merge_method=LoraMergeMethod.CUSTOM,
        default_width=768,
        default_height=768,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.processor = None
        self.video_vae = None
        self.audio_vae = None
        self._model_shim = None
        self._lora_path = None
        self._lora_scale = 1.0

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Override BasePipeline.load (the krea2.py:482 shape).

        Skips _enable_xformers / _apply_offload_mode: components are placed
        individually and self.pipe is not a DiffusionPipeline, so those would be
        no-ops at best.
        """
        if lora_paths:
            first = lora_paths[0]
            self._lora_path = (
                list(first.values())[0] if isinstance(first, dict) else first
            )
        self._lora_scale = float(lora_scale)
        logger.info(
            f"Loading {self.__class__.__name__} "
            f"lora={self._lora_path} scale={self._lora_scale}"
        )
        if self.offload_mode != "none":
            logger.warning(
                "offload_mode=%s is ignored by MiniMax-H3 (components are placed manually)",
                self.offload_mode,
            )
        self._load_pipeline()

    def _resolve_comfy_file(self, component: str) -> str:
        """Find a weight file locally, else download it.

        Simplified from minimax_h3.py:215-261: the inference service always
        works against a populated model cache (src/services/download_config.py
        pre-fetches these four files), so the model_kwargs override chain and
        the name_or_path-as-local-folder branch are dropped. The recursive
        search under the category folder is kept, because the cache layout puts
        files under a snapshot subdirectory.
        """
        rel_path = COMFY_FILES[component]
        filename = os.path.basename(rel_path)
        category = os.path.dirname(rel_path)

        roots = [r for r in (self.model_path, settings.models_path) if r]
        for root in roots:
            for rel in (rel_path, filename):
                candidate = os.path.join(root, rel)
                if os.path.exists(candidate):
                    return candidate
        for root in roots:
            found = self._find_file_recursive(os.path.join(root, category), filename)
            if found is not None:
                return found

        import huggingface_hub

        target = roots[0] if roots else None
        logger.info(f"Downloading {rel_path} from {COMFY_REPO}")
        return huggingface_hub.hf_hub_download(
            repo_id=COMFY_REPO, filename=rel_path, local_dir=target
        )

    @staticmethod
    def _find_file_recursive(root_dir: str, filename: str) -> Optional[str]:
        """First (breadth-stable, sorted) match anywhere under root_dir.
        minimax_h3.py:204-213."""
        if not os.path.isdir(root_dir):
            return None
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def _load_transformer(self):
        """Ported from minimax_h3.py:273-313.

        Every step here has a silent-wrong-output failure mode, so the
        load_state_dict result assertion is kept verbatim: it is the tripwire
        that turns a bad port into an exception instead of plausible garbage.
        """
        from safetensors.torch import load_file
        from toolkit.util.comfy_quant_import import (
            import_comfy_quantized_layers,
            OstrisLinear,
        )

        mods = _load_aitk_h3_modules()
        transformer_mod = mods["transformer"]

        dit_path = self._resolve_comfy_file("dit_fl2va")
        logger.info(f"Loading MiniMax-H3 transformer from {dit_path}")
        state_dict = load_file(dit_path)

        params = transformer_mod.MiniMaxH3TransformerParams()
        table = state_dict.get("adaln_t_table", None)
        if table is not None:
            # pruned checkpoint: factored timestep table instead of the MLP.
            # NOTE the SiLU is deliberately absent on the AdaLN path
            # (transformer.py:68-71) -- do not "fix" that.
            params.adaln_t_table_size = table.shape[0]
            params.time_embed_dim = table.shape[1]

        with torch.device("meta"):
            transformer = transformer_mod.MiniMaxH3Transformer(params)

        state_dict, num_quantized = import_comfy_quantized_layers(
            transformer, state_dict, orig_dtype=self.torch_dtype
        )
        if num_quantized:
            logger.info(f" - attached {num_quantized} pre-quantized ConvRot layers")

        result = transformer.load_state_dict(state_dict, assign=True, strict=False)
        quantized_weight_keys = {
            f"{name}.weight"
            for name, m in transformer.named_modules()
            if isinstance(m, OstrisLinear)
        }
        bad_missing = [k for k in result.missing_keys if k not in quantized_weight_keys]
        if bad_missing or result.unexpected_keys:
            raise ValueError(
                f"MiniMax-H3 transformer load mismatch: missing {bad_missing[:8]}, "
                f"unexpected {result.unexpected_keys[:8]}"
            )
        del state_dict
        _flush()
        return transformer

    def _load_text_encoder(self):
        """Ported from minimax_h3.py:315-412.

        The key_map, the 50-layer truncation, lm_head=None,
        tie_word_embeddings=False and the final-norm Identity are all
        load-bearing: only hidden_states[50] is consumed and it must be the
        UNnormalized layer-49 output.
        """
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import (
            AutoConfig,
            AutoProcessor,
            AutoTokenizer,
            Qwen3VLForConditionalGeneration,
        )
        from toolkit.util.comfy_quant_import import (
            import_comfy_quantized_layers,
            OstrisLinear,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            ORIGINAL_REPO, subfolder="FL2VA/tokenizer"
        )
        processor = AutoProcessor.from_pretrained(
            ORIGINAL_REPO, subfolder="FL2VA/processor"
        )
        if not hasattr(processor, "create_mm_token_type_ids"):
            raise RuntimeError(
                "This transformers build lacks Qwen3VLProcessor.create_mm_token_type_ids, "
                "which MiniMax-H3's prompt encoder calls unconditionally "
                "(minimax_h3/src/text_encoder.py:86). ai-toolkit pins "
                "transformers==5.5.3; the inference requirements may be older."
            )

        te_file = self._resolve_comfy_file("text_encoder")
        logger.info(f"Loading Qwen3-VL text encoder from {te_file}")

        config = AutoConfig.from_pretrained(ORIGINAL_REPO, subfolder="FL2VA/text_encoder")
        config.text_config.num_hidden_layers = TEXT_ENCODER_LAYER
        config.tie_word_embeddings = False
        with init_empty_weights():
            text_encoder = Qwen3VLForConditionalGeneration(config)
        text_encoder.lm_head = None

        state_dict = load_file(te_file)

        def key_map(prefix: str) -> str:
            if prefix.startswith("model."):
                return "model.language_model." + prefix[len("model.") :]
            if prefix.startswith("visual."):
                return "model." + prefix
            return prefix

        state_dict, num_quantized = import_comfy_quantized_layers(
            text_encoder,
            state_dict,
            orig_dtype=self.torch_dtype,
            key_map=key_map,
        )
        logger.info(f" - attached {num_quantized} pre-quantized nvfp4/int8 layers")

        state_dict = {
            key_map(k[: k.rfind(".")]) + k[k.rfind(".") :]: v
            for k, v in state_dict.items()
        }
        result = text_encoder.load_state_dict(state_dict, assign=True, strict=False)
        quantized_keys = {
            f"{name}.weight"
            for name, m in text_encoder.named_modules()
            if isinstance(m, OstrisLinear)
        }
        allowed_missing_prefixes = (
            "lm_head",
            "model.language_model.norm",
            "model.language_model.embed_tokens",
        )
        bad_missing = [
            k
            for k in result.missing_keys
            if k not in quantized_keys and not k.startswith(allowed_missing_prefixes)
        ]
        if bad_missing or result.unexpected_keys:
            raise ValueError(
                f"MiniMax-H3 text encoder load mismatch: missing {bad_missing[:8]}, "
                f"unexpected {result.unexpected_keys[:8]}"
            )
        del state_dict

        text_encoder.model.language_model.norm = torch.nn.Identity()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        _flush()
        return tokenizer, processor, text_encoder

    def _load_vaes(self):
        """Ported from minimax_h3.py:414-449.

        latents_mean / latents_std ride along in the comfy files and must be
        restored as float32 into the modules' non-persistent buffers.
        """
        from safetensors.torch import load_file

        mods = _load_aitk_h3_modules()
        vae_mod, audio_vae_mod = mods["vae"], mods["audio_vae"]

        logger.info("Loading MiniMax-H3 video VAE")
        video_sd = load_file(self._resolve_comfy_file("video_vae"))
        video_stats = {
            k: video_sd.pop(k).float()
            for k in ("latents_mean", "latents_std")
            if k in video_sd
        }
        video_vae = vae_mod.MiniMaxH3VideoVAE()
        video_vae.load_state_dict(video_sd, strict=True, assign=True)
        for k, v in video_stats.items():
            getattr(video_vae, k).copy_(v)
        video_vae.eval().requires_grad_(False)
        del video_sd

        logger.info("Loading MiniMax-H3 audio VAE")
        audio_sd = load_file(self._resolve_comfy_file("audio_vae"))
        audio_stats = {
            k: audio_sd.pop(k).float()
            for k in ("latents_mean", "latents_std")
            if k in audio_sd
        }
        # the comfy repack ships weight norm already folded; only fold when the
        # raw parametrization is present (original-repo file)
        if any(k.endswith("weight_g") for k in audio_sd.keys()):
            audio_sd = audio_vae_mod.fold_audio_vae_weight_norm(audio_sd)
        audio_vae = audio_vae_mod.MiniMaxH3AudioVAE()
        audio_vae.load_state_dict(audio_sd, strict=True, assign=True)
        for k, v in audio_stats.items():
            getattr(audio_vae, k).copy_(v)
        audio_vae.to(torch.float32).eval().requires_grad_(False)
        del audio_sd
        _flush()
        return video_vae, audio_vae

    def _attach_lora(self, transformer, lora_path: str, lora_scale: float) -> int:
        """Attach a live LoRA adapter to every matching linear. Returns count.

        Key handling follows ai-toolkit's save path:
          * `diffusion_model.` prefix is stripped (minimax_h3.py:959-970)
          * H3 is is_transformer=True, so lora_special.py:422-431 forces
            peft_format -- keys are lora_A/lora_B and the `.alpha` tensors are
            STRIPPED (network_mixins.py:605-614) with alpha == rank. A missing
            alpha therefore defaults to rank, NOT to 1. Getting this wrong is
            the trap krea2.py:695-697 documents.

        Verified against a real rank-16 adapter from job a1977b6b (516 tensors):
        258 complete pairs, 0 alpha tensors, 0 unmatched keys. The targets are

            50x  blocks.N.adaln_proj.linear
            50x  blocks.N.attn.qkv_proj
            50x  blocks.N.attn.out_proj
            50x  blocks.N.mlp.fc1
            50x  blocks.N.mlp.fc2
             8x  token_refiner.blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}

        Note adaln_proj and the token_refiner subtree: a `startswith("blocks.")`
        whitelist would silently drop 58 of 258 modules (23%) and still "work".
        That is why this matches against transformer.named_modules() with no
        prefix filter and raises when nothing attaches -- krea2.py hit exactly
        that failure with a whitelist dropping 32/256.
        """
        from safetensors.torch import load_file

        sd = load_file(lora_path)

        pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        alphas: Dict[str, float] = {}
        for key, tensor in sd.items():
            name = key
            if name.startswith("diffusion_model."):
                name = name[len("diffusion_model.") :]
            if name.endswith(".alpha"):
                alphas[name[: -len(".alpha")]] = float(tensor.item())
                continue
            for suffix, slot in (
                (".lora_A.default.weight", "down"),
                (".lora_B.default.weight", "up"),
                (".lora_A.weight", "down"),
                (".lora_B.weight", "up"),
                (".lora_down.weight", "down"),
                (".lora_up.weight", "up"),
            ):
                if name.endswith(suffix):
                    pairs.setdefault(name[: -len(suffix)], {})[slot] = tensor
                    break

        modules = dict(transformer.named_modules())
        attached = 0
        skipped: List[str] = []
        for module_name, wt in pairs.items():
            if "down" not in wt or "up" not in wt:
                skipped.append(f"{module_name} (incomplete pair)")
                continue
            target = modules.get(module_name)
            if target is None:
                skipped.append(f"{module_name} (no such module)")
                continue
            rank = wt["down"].shape[0]
            alpha = alphas.get(module_name, rank)  # peft: alpha == rank
            scale = (alpha / rank) * lora_scale
            parent_name, _, attr = module_name.rpartition(".")
            parent = modules.get(parent_name) if parent_name else transformer
            if parent is None:
                skipped.append(f"{module_name} (no parent)")
                continue
            device = next(
                (p.device for p in target.parameters(recurse=False)), self.device
            )
            setattr(
                parent,
                attr,
                _H3LoraLinear(
                    target,
                    wt["down"].to(device, torch.float32),
                    wt["up"].to(device, torch.float32),
                    scale,
                ),
            )
            attached += 1

        if attached == 0:
            raise ValueError(
                f"MiniMax-H3 LoRA {lora_path!r} attached to 0 modules. "
                f"Parsed {len(pairs)} candidate pairs. First few skipped: {skipped[:5]}"
            )
        if skipped:
            logger.warning(
                "MiniMax-H3 LoRA: attached %d modules, skipped %d (%s)",
                attached,
                len(skipped),
                "; ".join(skipped[:5]),
            )
        else:
            logger.info("MiniMax-H3 LoRA: attached %d modules at scale %.3f",
                        attached, lora_scale)
        return attached

    def _load_pipeline(self):
        mods = _load_aitk_h3_modules()
        pipeline_mod, packing = mods["pipeline"], mods["packing"]

        self.transformer = self._load_transformer()
        self.tokenizer, self.processor, self.text_encoder = self._load_text_encoder()
        self.video_vae, self.audio_vae = self._load_vaes()

        if self._lora_path:
            self._attach_lora(self.transformer, self._lora_path, self._lora_scale)

        device = self.device
        self.transformer.to(device)
        self.text_encoder.to(device)
        self.video_vae.to(device)
        self.audio_vae.to(device)

        self._model_shim = _ModelShim(
            transformer=self.transformer,
            video_vae=self.video_vae,
            audio_vae=self.audio_vae,
            device=torch.device(device),
            dtype=self.torch_dtype,
            packing=packing,
        )
        self.pipe = pipeline_mod.MiniMaxH3Pipeline(self._model_shim)
        _flush()

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str, control_image: Optional[Image.Image]):
        """Return the _EmbedsShim the sampler expects.

        pipeline.py:100-101 reads conditional_embeds.text_embeds[0] and
        .text_token_tags[0] only, so a SimpleNamespace with two 1-element lists
        is the whole contract (minimax_h3.py:522-575 builds the real
        AdvancedPromptEmbeds the same way).
        """
        mods = _load_aitk_h3_modules()
        encode = mods["text_encoder"].encode_minimax_h3_prompt

        keyframes = [control_image] if control_image is not None else None
        embeds, tags = encode(
            self.text_encoder,
            self.tokenizer,
            self.processor,
            prompt.strip(),
            keyframes=keyframes,
            device=torch.device(self.device),
            dtype=self.torch_dtype,
        )
        return SimpleNamespace(text_embeds=[embeds], text_token_tags=[tags])

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
        num_frames: int = 107,
        fps: int = FPS,
        output_type: str = "pil",
        latents: Optional[torch.Tensor] = None,
        denoise_strength: Optional[float] = None,
        resolution: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Wide signature on purpose.

        base.py:822-859 calls with output_type/latents/denoise_strength/
        resolution inside a try/except TypeError. A narrow signature makes every
        call raise-and-retry, and worse, a genuine TypeError raised inside the
        model forward would be swallowed and the whole generation silently
        re-run. krea2.py:885-898 has that shape; this does not.
        """
        mods = _load_aitk_h3_modules()
        packing = mods["packing"]

        if latents is not None or output_type != "pil":
            raise NotImplementedError(
                "MiniMax-H3 does not support latents input or non-pil output_type"
            )

        if num_frames <= 1:
            # pipeline.py:198-199 returns [PIL.Image] rather than the dict in
            # this mode, which the video-saving path cannot consume. Reject
            # explicitly rather than crash downstream.
            raise ValueError(
                "MiniMax-H3 single-frame (image) mode is not exposed; "
                "use num_frames on the 17n+5 grid (5, 22, 39, ..., 107, 124)"
            )

        requested_frames = num_frames
        num_frames = packing.align_num_frames_down(num_frames)
        if num_frames != requested_frames:
            logger.info(
                "MiniMax-H3: snapped num_frames %d -> %d (17n+5 grid)",
                requested_frames,
                num_frames,
            )

        ctrl = control_image
        if ctrl is None and control_images:
            ctrl = control_images[0]
        if ctrl is not None:
            ctrl = packing.prepare_keyframe_image(ctrl, height, width, stretch=True)

        conditional_embeds = self._encode_prompt(prompt, ctrl)

        out = self.pipe(
            conditional_embeds=conditional_embeds,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            ctrl_img=ctrl,
            with_audio=True,
        )

        # key rename: the sampler returns "video", executor._save_video_result
        # (executor.py:379-382) reads "video_tensor". out["video"] is already
        # [T,H,W,C] uint8 on cpu (pipeline.py:196-197), which is exactly what
        # encode_video unpacks -- do NOT route it through `frames=`, which
        # permutes to [T,C,H,W] (image_utils.py:213-215).
        return {
            "video_tensor": out["video"],
            "fps": out.get("fps", FPS),
            "audio": out.get("audio"),
            "audio_sample_rate": out.get("audio_sample_rate", AUDIO_SAMPLE_RATE),
        }
