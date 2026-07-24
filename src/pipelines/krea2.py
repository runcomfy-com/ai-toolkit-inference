"""
Krea 2 pipelines (raw / turbo / edit / edit-turbo).

Krea 2 is a single-stream MMDiT text-to-image model:
  - denoiser:     SingleStreamDiT (~12.8B params), ai-toolkit
                  extensions_built_in/diffusion_models/krea2/src/mmdit.py
  - text encoder: Qwen3-VL-4B-Instruct, a stack of 12 hidden-state layers
  - autoencoder:  the Qwen-Image VAE (f8, 16 latent channels)

There is no diffusers pipeline for Krea 2 at the diffusers commit this repo pins,
and ai-toolkit's own Krea2Pipeline is not a DiffusionPipeline (no
load_lora_weights / fuse_lora / offload hooks, and no callback_on_step_end). So
this wrapper:
  * builds the three components by hand,
  * merges the user LoRA into the transformer manually (LoraMergeMethod.CUSTOM),
  * runs the flow-matching Euler loop itself, importing ai-toolkit's
    predict_velocity() / timesteps() / pad_text_features() / encode_krea_prompt()
    so every subtle part (mu schedule, 2x2 packing, reference-latent RoPE
    placement, ref K/V cache, prompt template) stays upstream and cannot drift
    from what the trainer used to render its preview samples.

IMPORT SAFETY: every heavy import lives inside a method. src/api/v1/inference.py
imports PIPELINE_REGISTRY at module scope and src/pipelines/__init__.py builds it
with no try/except, so a top-level import failure here would break FastAPI
startup for every model in the catalog.
"""

import gc
import logging
import math
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

# ai-toolkit extensions_built_in/diffusion_models/krea2/krea2.py:64-77
KREA2_MMDIT_CONFIG = dict(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

QWEN3_VL_PATH = "Qwen/Qwen3-VL-4B-Instruct"  # krea2.py:97
QWEN_IMAGE_VAE_PATH = "Qwen/Qwen-Image"  # krea2.py:98

VAE_SCALE_FACTOR = 8  # krea2.py:186, the Qwen-Image VAE is f8
MAX_TEXT_LENGTH = 512  # krea2.py:189-191
VLM_MAX_PIXELS = 384 * 384  # krea2.py:686
CONTROL_IMAGE_MAX_PIXELS = 1024 * 1024  # krea2.py:552-555
MAX_CONTROL_IMAGES = 3  # krea2.py:504-513

# Timestep-schedule endpoints (krea2/src/pipeline.py:313-347). `mu` is left None
# so it is interpolated from the image-token count, which is what the trainer
# does: no krea2 preset pins model_kwargs.schedule_mu, not even the distilled
# turbo ones. Krea's own CLI passes --mu 1.15 for turbo; we deliberately follow
# the trainer instead, because this repo's product is training-preview parity.
SCHEDULE = {"y1": 0.5, "y2": 1.15, "min_res": 256, "max_res": 1280, "mu": None}


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_AITK_MODULES: Optional[Dict[str, Any]] = None

_KREA2_REL_PATH = os.path.join("extensions_built_in", "diffusion_models", "krea2", "src")


def _find_krea2_src_dir() -> str:
    """Locate ai-toolkit's krea2/src directory.

    Tries, in order:

    1. `settings.ai_toolkit_path` -- AI_TOOLKIT_PATH env var, else
       <repo>/vendor/ai-toolkit, else /app/ai-toolkit (config.py's own chain,
       whose last step already matches the production image layout).
    2. Wherever `extensions_built_in` already resolves on sys.path. The
       production image (Images/ai-toolkit-inference/Dockerfile) exposes
       ai-toolkit through PYTHONPATH rather than AI_TOOLKIT_PATH, so this covers
       a checkout mounted somewhere other than the three paths above.
       find_spec() locates the package without executing its __init__.

    Raises ImportError listing everything that was tried.
    """
    import importlib.util

    tried = []

    def _try(candidate: str) -> bool:
        if candidate in tried:
            return False
        tried.append(candidate)
        return os.path.isdir(candidate)

    configured = settings.ai_toolkit_path
    if configured:
        candidate = os.path.join(configured, _KREA2_REL_PATH)
        if _try(candidate):
            return candidate

    try:
        spec = importlib.util.find_spec("extensions_built_in")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidate = os.path.join(location, "diffusion_models", "krea2", "src")
            if _try(candidate):
                return candidate

    raise ImportError(
        "Krea 2 requires an ostris/ai-toolkit checkout containing "
        "extensions_built_in/diffusion_models/krea2/ (ai-toolkit >= v0.11.0). "
        "Tried:\n  " + "\n  ".join(tried) + "\n"
        "Set AI_TOOLKIT_PATH, put ai-toolkit on PYTHONPATH, or run install.py."
    )


def _load_aitk_krea2_modules() -> Dict[str, Any]:
    """Load ai-toolkit's krea2 leaf modules by file path.

    Deliberately does NOT go through `import extensions_built_in.diffusion_models
    .krea2...`: that package's __init__ eagerly imports all ~21 model
    architectures, dragging in av / torchaudio / optimum.quanto / toolkit.*,
    none of which Krea 2 needs. The three modules loaded here import only torch,
    einops, PIL and one diffusers helper, so loading them directly keeps this
    pipeline usable in a slim runtime and avoids a large chunk of import cost.

    This still executes upstream's own source -- nothing is vendored or
    reimplemented -- so the sampling math cannot drift from the trainer.
    """
    global _AITK_MODULES
    if _AITK_MODULES is not None:
        return _AITK_MODULES

    import importlib.util
    from types import ModuleType

    src_dir = _find_krea2_src_dir()

    # A synthetic parent package so `from .mmdit import ...` inside pipeline.py
    # resolves without touching the real extensions_built_in package.
    pkg_name = "_aitk_krea2_src"
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [src_dir]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    modules: Dict[str, Any] = {}
    # mmdit first: pipeline.py imports it relatively.
    for name in ("mmdit", "text_encoder", "pipeline"):
        full_name = f"{pkg_name}.{name}"
        if full_name in sys.modules:
            modules[name] = sys.modules[full_name]
            continue
        path = os.path.join(src_dir, f"{name}.py")
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Krea 2 module from {path!r}")
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


def _patch_qwen_vl_patch_embed(model) -> int:
    """Swap Qwen-VL's Conv3d vision patch_embed for the equivalent F.linear.

    Inlined from ai-toolkit krea2.py:103-125 rather than imported, because
    importing krea2.py would pull in toolkit.* and optimum.quanto. The Conv3d
    has kernel == stride, i.e. it is a plain linear projection of each flattened
    patch, and bf16 Conv3d has no fast cuDNN kernel. The weight is read lazily so
    this survives later .to(device)/dtype moves.
    """
    import torch.nn.functional as F

    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(
            proj.stride
        ):

            def fast_forward(hidden_states, _proj=proj):
                w = _proj.weight.reshape(_proj.weight.shape[0], -1)
                x = hidden_states.view(-1, w.shape[1]).to(w.dtype)
                return F.linear(x, w, _proj.bias)

            # Replaces the PARENT module's forward (krea2.py:122), not proj's.
            module.forward = fast_forward
            patched += 1
    return patched


class _Krea2Sampler:
    """Diffusers-shaped wrapper around the Krea 2 flow-matching loop.

    A port of extensions_built_in/diffusion_models/krea2/src/pipeline.py:292-392
    with a step callback added. All of the subtle math (predict_velocity,
    timesteps, pad_text_features) is imported from ai-toolkit, never
    reimplemented here.

    Why not call ai-toolkit's Krea2Pipeline directly: it takes a live Krea2Model
    and reads .model_config.model_kwargs / .kv_cache / .decode_latents off it,
    and its __call__ has no callback parameter -- so the ComfyUI progress bar and
    cancel button would silently do nothing.
    """

    def __init__(
        self,
        transformer,
        vae,
        device,
        dtype,
        patch_size,
        vae_scale_factor,
        kv_cache,
        schedule,
    ):
        self.transformer = transformer
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.patch_size = patch_size
        self.vae_scale_factor = vae_scale_factor
        self.kv_cache = kv_cache
        self.schedule = schedule

    def to(self, *args, **kwargs):
        # Components are placed individually in _load_pipeline().
        return self

    # ------------------------------------------------------------------ VAE

    def _mean_std(self, ref: torch.Tensor):
        """Qwen-Image VAE per-channel latent normalization (krea2.py:795-802)."""
        cfg = self.vae.config
        mean = (
            torch.tensor(cfg.latents_mean)
            .view(1, cfg.z_dim, 1, 1, 1)
            .to(ref.device, ref.dtype)
        )
        std = (
            torch.tensor(cfg.latents_std)
            .view(1, cfg.z_dim, 1, 1, 1)
            .to(ref.device, ref.dtype)
        )
        return mean, std

    def encode_images(self, images: torch.Tensor, generator=None) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1] -> (B, 16, H/8, W/8) normalized latents."""
        images = images.to(self.device, dtype=self.dtype).unsqueeze(2)  # video-VAE frame dim
        posterior = self.vae.encode(images).latent_dist
        # Training samples the posterior off the global RNG (krea2.py:793), which
        # a stateless server cannot reproduce. Threading the request's own CPU
        # generator keeps the same distribution and makes it reproducible.
        latents = posterior.sample(generator=generator)
        mean, std = self._mean_std(latents)
        return ((latents - mean) * (1.0 / std)).squeeze(2)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.device, dtype=self.dtype).unsqueeze(2)
        mean, std = self._mean_std(latents)
        return self.vae.decode(latents * std + mean).sample.squeeze(2)

    # ------------------------------------------------------------- sampling

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds: List[torch.Tensor],
        unconditional_embeds: Optional[List[torch.Tensor]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,  # ALREADY 0-normalized (user cfg - 1)
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        ref_latents: Optional[List[List[torch.Tensor]]] = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        from diffusers.utils.torch_utils import randn_tensor

        aitk = _load_aitk_krea2_modules()["pipeline"]
        pad_text_features = aitk.pad_text_features
        predict_velocity = aitk.predict_velocity
        timesteps = aitk.timesteps

        device, dtype = self.device, self.dtype
        ae, patch = self.vae_scale_factor, self.patch_size
        gh, gw = height // (ae * patch), width // (ae * patch)

        if latents is None:
            latents = randn_tensor(
                (1, self.transformer.config.channels, height // ae, width // ae),
                generator=generator,
                device=torch.device(device),
                dtype=torch.float32,
            )
        latents = latents.to(device, dtype=torch.float32)

        do_cfg = guidance_scale > 0 and unconditional_embeds is not None
        cond_feats, cond_mask = pad_text_features(conditional_embeds, device, dtype)
        if do_cfg:
            uncond_feats, uncond_mask = pad_text_features(
                unconditional_embeds, device, dtype
            )

        s, align = self.schedule, ae * patch
        ts = timesteps(
            gh * gw,
            num_inference_steps,
            (int(s["min_res"]) // align) ** 2,  # x1 = 256
            (int(s["max_res"]) // align) ** 2,  # x2 = 6400
            y1=float(s["y1"]),
            y2=float(s["y2"]),
            mu=(float(s["mu"]) if s.get("mu") is not None else None),
        )

        # With kv_cache (isolated ref attention) the reference K/V are
        # step-invariant, so the first model call doubles as the precompute pass.
        isolate = self.kv_cache
        ref_cache = None
        if isolate and ref_latents and any(len(r) > 0 for r in ref_latents):
            ref_cache = {"kv": None, "mask": None}

        for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
            t = torch.full((latents.shape[0],), tcurr, dtype=dtype, device=device)
            v = predict_velocity(
                self.transformer,
                latents.to(dtype),
                t,
                cond_feats,
                cond_mask,
                ref_latents=ref_latents,
                isolate_refs=isolate,
                ref_kv_cache=ref_cache,
            )
            if do_cfg:
                v_uncond = predict_velocity(
                    self.transformer,
                    latents.to(dtype),
                    t,
                    uncond_feats,
                    uncond_mask,
                    ref_latents=ref_latents,
                    isolate_refs=isolate,
                    ref_kv_cache=ref_cache,
                )
                v = v + guidance_scale * (v - v_uncond)
            # Latent state stays fp32 while the model runs in bf16 (pipeline.py:386).
            latents = latents + (tprev - tcurr) * v.to(torch.float32)

            if callback_on_step_end is not None:
                out = callback_on_step_end(self, i, tcurr, {"latents": latents})
                if isinstance(out, dict):
                    latents = out.pop("latents", latents)

        images = self.decode_latents(latents).float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        arr = images.permute(0, 2, 3, 1).cpu().numpy()
        return SimpleNamespace(images=[Image.fromarray(a) for a in arr])


class Krea2Pipeline(BasePipeline):
    """Krea 2 (raw), text-to-image. Base class for all four Krea 2 arches."""

    CONFIG = PipelineConfig(
        model_type=ModelType.KREA2,
        base_model="krea/Krea-2-Raw",
        resolution_divisor=16,  # vae_scale_factor(8) * patch(2), krea2.py:226-228
        default_steps=30,  # ai-toolkit ui/src/helpers/defaultSamples.ts:48
        default_guidance_scale=4.0,  # ai-toolkit ui/src/helpers/defaultSamples.ts:47
        requires_control_image=False,
        supports_negative_prompt=True,
        lora_merge_method=LoraMergeMethod.CUSTOM,
        default_width=1024,
        default_height=1024,
    )

    # ---- per-arch knobs (the flux2_klein.py subclassing pattern) ----
    # krea2.py:148-153 derives the checkpoint filename from the repo-id tail:
    # "krea/Krea-2-Raw" -> "raw.safetensors".
    CHECKPOINT_FILENAME = "raw.safetensors"
    IS_EDIT = False
    USE_KV_CACHE = False
    MATCH_TARGET_RES = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.suffix_tokenizer = None  # Qwen2TokenizerFast, suffix-only pass
        self.vl_processor = None  # Qwen3-VL AutoProcessor, edit only
        self._lora_path = None
        self._lora_scale = 1.0
        # Per-request overrides for the two edit settings that must match how the
        # LoRA was trained. None => use the class default. See
        # apply_model_options() for why these cannot be auto-detected.
        self._kv_cache_override: Optional[bool] = None
        self._match_target_res_override: Optional[bool] = None

    @property
    def use_kv_cache(self) -> bool:
        if self._kv_cache_override is not None:
            return self._kv_cache_override
        return self.USE_KV_CACHE

    @property
    def match_target_res(self) -> bool:
        if self._match_target_res_override is not None:
            return self._match_target_res_override
        return self.MATCH_TARGET_RES

    def apply_model_options(self, **options) -> None:
        """Accept per-request `kv_cache` / `match_target_res` overrides.

        Both must match the values the LoRA was TRAINED with, and neither is
        recoverable from the adapter file -- ai-toolkit's LoRA metadata carries
        only training_info / ss_base_model_version / ss_output_name / the
        trigger word, never model_kwargs.

        This matters in practice: ai-toolkit's krea2 edit preset gained
        `kv_cache: true` + `match_target_res: true` on 2026-07-16. Adapters
        trained before that ran with kv_cache off and match_target_res off, and
        `kv_cache` is not an optional speed-up -- it changes the attention mask,
        so a mismatch silently produces wrong output rather than an error.

        The class defaults follow the current preset, so callers that say
        nothing get the right behavior for newly trained adapters.

        Each call is AUTHORITATIVE for the whole option set: an option that is
        absent resets to the class default rather than inheriting whatever the
        previous request set. The pipeline cache hands the same instance to
        every request, so anything else would leak one request's settings into
        the next.
        """
        kv_cache = options.get("kv_cache")
        self._kv_cache_override = None if kv_cache is None else bool(kv_cache)
        match_target_res = options.get("match_target_res")
        self._match_target_res_override = (
            None if match_target_res is None else bool(match_target_res)
        )

        # The sampler reads kv_cache per call, so this takes effect immediately
        # with no reload.
        if self.pipe is not None:
            self.pipe.kv_cache = self.use_kv_cache

    # ------------------------------------------------------------------ load

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Override BasePipeline.load (the chroma.py / flux2.py shape).

        Skips _enable_xformers / _apply_offload_mode: components are placed
        individually below and self.pipe is not a DiffusionPipeline, so those
        would be no-ops at best.
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
                "offload_mode=%s is ignored by Krea 2 (components are placed manually)",
                self.offload_mode,
            )
        self._load_pipeline()

    def _import_aitk(self):
        mmdit = _load_aitk_krea2_modules()["mmdit"]
        return mmdit.SingleMMDiTConfig, mmdit.SingleStreamDiT

    def _load_pipeline(self):
        import huggingface_hub
        from diffusers import AutoencoderKLQwenImage
        from safetensors.torch import load_file
        from transformers import (
            AutoTokenizer,
            Qwen2TokenizerFast,
            Qwen3VLForConditionalGeneration,
        )

        SingleMMDiTConfig, SingleStreamDiT = self._import_aitk()
        if self.IS_EDIT:
            self._assert_edit_capable()
        _flush()

        # -- 1. transformer: meta build then materialize from the checkpoint ---
        logger.info(f"[1/4] Building SingleStreamDiT ({self.CONFIG.base_model})")
        with torch.device("meta"):
            transformer = SingleStreamDiT(SingleMMDiTConfig(**KREA2_MMDIT_CONFIG))

        ckpt = huggingface_hub.hf_hub_download(
            repo_id=self.CONFIG.base_model,
            filename=self.CHECKPOINT_FILENAME,
            token=self.hf_token,
        )
        state_dict = load_file(ckpt, device="cpu")
        state_dict = {
            k: (v.to(self.dtype) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }
        transformer.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        transformer.to(self.device, dtype=self.dtype)
        transformer.requires_grad_(False)
        transformer.eval()
        _flush()

        # -- 2. LoRA merged into the base weights -----------------------------
        if self._lora_path and os.path.exists(self._lora_path):
            logger.info(
                f"[2/4] Merging LoRA {self._lora_path} scale={self._lora_scale}"
            )
            self._merge_lora_to_transformer(
                transformer, self._lora_path, self._lora_scale
            )
            self.lora_loaded = True
        elif self._lora_path:
            logger.error(f"LoRA file not found: {self._lora_path}")
        else:
            logger.warning("No LoRA path provided; running base weights only")

        # -- 3. Qwen3-VL text encoder (krea2.py:260-289) ----------------------
        logger.info(f"[3/4] Loading text encoder {QWEN3_VL_PATH} (edit={self.IS_EDIT})")
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN3_VL_PATH, max_length=MAX_TEXT_LENGTH, token=self.hf_token
        )
        suffix_tokenizer = Qwen2TokenizerFast.from_pretrained(
            QWEN3_VL_PATH, max_length=MAX_TEXT_LENGTH, token=self.hf_token
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN3_VL_PATH, torch_dtype=self.dtype, token=self.hf_token
        )
        vl_processor = None
        if self.IS_EDIT:
            from transformers import AutoProcessor

            vl_processor = AutoProcessor.from_pretrained(
                QWEN3_VL_PATH, token=self.hf_token
            )
            _patch_qwen_vl_patch_embed(text_encoder)
        elif getattr(text_encoder.model, "visual", None) is not None:
            # Text-to-image never uses the vision tower -- krea2.py:281-285.
            text_encoder.model.visual = None
        text_encoder.to(self.device, dtype=self.dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        self._check_prompt_prefix_length(tokenizer)
        _flush()

        # -- 4. Qwen-Image VAE (krea2.py:291-299) -----------------------------
        logger.info(f"[4/4] Loading VAE {QWEN_IMAGE_VAE_PATH}")
        vae = AutoencoderKLQwenImage.from_pretrained(
            QWEN_IMAGE_VAE_PATH,
            subfolder="vae",
            torch_dtype=self.dtype,
            token=self.hf_token,
        )
        vae.to(self.device, dtype=self.dtype)
        vae.requires_grad_(False)
        vae.eval()

        self.transformer, self.vae = transformer, vae
        self.text_encoder, self.tokenizer = text_encoder, tokenizer
        self.suffix_tokenizer, self.vl_processor = suffix_tokenizer, vl_processor

        self.pipe = _Krea2Sampler(
            transformer=transformer,
            vae=vae,
            device=self.device,
            dtype=self.dtype,
            patch_size=KREA2_MMDIT_CONFIG["patch"],
            vae_scale_factor=VAE_SCALE_FACTOR,
            kv_cache=self.use_kv_cache,
            schedule=dict(SCHEDULE),
        )

        _flush()
        if torch.cuda.is_available():
            logger.info(
                f"GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated"
            )
        logger.info(f"{self.__class__.__name__} loaded")

    # ---------------------------------------------------------------- guards

    def _assert_edit_capable(self):
        """Edit conditioning drives Qwen3-VL M-RoPE through mm_token_type_ids.

        In an older transformers, Qwen3-VL derives vision positions from
        (input_ids, image_grid_thw, video_grid_thw, attention_mask) and swallows
        mm_token_type_ids through **kwargs -- producing wrong conditioning with
        no error at all. Fail loudly instead of silently mis-conditioning.

        Version landscape (checked 2026-07-23):
          - ai-toolkit pins transformers==5.5.3, which is what krea2 edit was
            trained against.
          - The production image (Images/ai-toolkit-inference/Dockerfile)
            installs ai-toolkit's requirements and then only this repo's
            requirements.txt (web deps), so the container gets 5.5.3 and edit
            works there.
          - requirements-inference.txt -- the LOCAL dev install -- pins 4.57.3,
            so a local run may trip this guard even though the deployment is fine.
        """
        import inspect

        import transformers
        from transformers.models.qwen3_vl import modeling_qwen3_vl as M

        ok = "mm_token_type_ids" in inspect.signature(
            M.Qwen3VLModel.forward
        ).parameters or "mm_token_type_ids" in inspect.signature(
            M.Qwen3VLModel.get_rope_index
        ).parameters
        if not ok:
            raise RuntimeError(
                f"{self.CONFIG.model_type.value} needs a transformers build whose "
                "Qwen3-VL M-RoPE consumes mm_token_type_ids (ai-toolkit trains "
                f"against 5.5.3). Installed: {transformers.__version__}. "
                "Text-to-image krea2 / krea2_turbo are unaffected."
            )

    def _check_prompt_prefix_length(self, tokenizer):
        """PROMPT_TEMPLATE_ENCODE_START_IDX = 34 is hardcoded in ai-toolkit but
        never checked against a shipped tokenizer. An off-by-N slices the wrong
        tokens off every conditioning vector, silently."""
        te = _load_aitk_krea2_modules()["text_encoder"]
        try:
            n = len(tokenizer(te.PROMPT_TEMPLATE_ENCODE_PREFIX).input_ids)
        except Exception as e:  # pragma: no cover - tokenizer quirks
            logger.debug(f"Could not verify prompt prefix length: {e}")
            return
        if n != te.PROMPT_TEMPLATE_ENCODE_START_IDX:
            logger.error(
                "Krea 2 prompt prefix tokenizes to %d tokens but the feature slice "
                "assumes %d -- conditioning will be wrong. Tokenizer mismatch.",
                n,
                te.PROMPT_TEMPLATE_ENCODE_START_IDX,
            )

    # ------------------------------------------------------------------ LoRA

    def _merge_lora_to_transformer(self, transformer, lora_path: str, lora_scale: float = 1.0):
        """Merge an ai-toolkit-trained krea2 LoRA into the transformer weights.

        Keys on disk are 'diffusion_model.<SingleStreamDiT module path>.lora_{A,B}.weight'
        (krea2.py:870-874 rewrites 'transformer.' -> 'diffusion_model.' on save).
        We merge into a bare SingleStreamDiT, so the prefix is stripped to empty.

        The loop is key-driven against transformer.state_dict() with NO prefix
        whitelist. ai-toolkit's module gate is a substring test for "blocks"
        (lora_special.py) and the text tower's submodules are literally named
        txtfusion.layerwise_blocks / txtfusion.refiner_blocks, so a
        startswith("blocks.") filter would silently drop 32 of 256 modules on a
        default-config LoRA.

        ai-toolkit forces peft_format for krea2, which means alpha == rank and no
        .alpha tensors are written -- a missing alpha MUST therefore default to
        rank (internal scale exactly 1.0), not to 1.
        """
        from safetensors.torch import load_file

        param_device = next(transformer.parameters()).device
        raw = load_file(lora_path, device="cpu")

        converted = {}
        for k, v in raw.items():
            nk = k.replace("diffusion_model.", "")
            nk = nk.replace(".lora_A.default.weight", ".lora_A.weight")
            nk = nk.replace(".lora_B.default.weight", ".lora_B.weight")
            converted[nk] = v.to(self.dtype)
        if param_device.type == "cuda":
            converted = {
                k: v.to(device=param_device, dtype=self.dtype, non_blocking=True)
                for k, v in converted.items()
            }

        diff_keys = [k for k in converted if k.endswith(".diff") or k.endswith(".diff_b")]
        if diff_keys:
            logger.warning(
                "Ignoring %d full-weight (.diff) keys; network.all_layers adapters "
                "are not supported",
                len(diff_keys),
            )

        if any("lokr_w" in k for k in converted):
            # krea2.py:197 sets use_old_lokr_format=False, so the keys are exactly
            # the peft-style lokr_* names flux2 already knows how to merge.
            logger.info("Detected LoKR adapter format")
            from .flux2 import Flux2Pipeline

            Flux2Pipeline._merge_lokr_to_transformer(
                self, transformer, converted, lora_scale
            )
            return

        rank = next(
            (converted[k].shape[0] for k in converted if "lora_A.weight" in k), None
        )
        if not rank:
            logger.error("No lora_A.weight in %s; nothing merged", lora_path)
            return

        state = transformer.state_dict()
        merged = skipped = 0
        for key in list(converted.keys()):
            if "lora_A.weight" not in key:
                continue
            base = key.replace(".lora_A.weight", "")
            b_key = key.replace("lora_A.weight", "lora_B.weight")
            if b_key not in converted:
                continue
            weight_key = base + ".weight"
            if weight_key not in state:
                skipped += 1
                continue
            alpha_t = converted.get(base + ".alpha")
            alpha = float(alpha_t.item()) if alpha_t is not None else rank
            scale = (alpha / rank) * lora_scale
            w = state[weight_key]
            # W += scale * (B @ A)
            w.addmm_(
                converted[b_key].to(w.device, w.dtype),
                converted[key].to(w.device, w.dtype),
                beta=1.0,
                alpha=scale,
            )
            merged += 1

        logger.info(
            "Merged %d LoRA layers with scale=%s (skipped %d)", merged, lora_scale, skipped
        )
        if merged == 0:
            raise ValueError(
                f"Merged 0 LoRA layers from {lora_path}. A default krea2 LoRA has "
                "256 modules (28 blocks x 8 + txtfusion 4 x 8). Check the file is "
                "a krea2 LoRA (metadata ss_base_model_version == 'krea2')."
            )
        if merged < 200:
            logger.warning(
                "Only %d LoRA layers merged (a default krea2 LoRA has 256)", merged
            )

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """LoRA is merged during _load_pipeline (the flux2.py shape)."""
        if not self.lora_loaded:
            logger.warning("_load_lora called but Krea 2 merges during _load_pipeline()")

    # -------------------------------------------------- reference-image path

    def _control_pils(self, control_image, control_images) -> List[Image.Image]:
        """Normalize the executor's control-image output to a list.

        executor._load_control_images passes a scalar for exactly one source and
        a list for more than one. Order matches krea2.py:504-513.
        """
        imgs = (
            list(control_images)
            if control_images
            else ([control_image] if control_image is not None else [])
        )
        return [im.convert("RGB") for im in imgs[:MAX_CONTROL_IMAGES]]

    def _prep_vlm_images(self, pils) -> List[torch.Tensor]:
        """Port of krea2.py:677-707: <=384^2 area, bicubic + antialias, min side
        28, never upscaled, kept float in [0, 1].

        Deliberately a DIFFERENT resize policy from _encode_ref_latents -- each
        reference image is consumed twice, once by the MLLM and once by the VAE.
        """
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor

        out = []
        for im in pils:
            t = to_tensor(im).to(self.device)
            h, w = t.shape[1], t.shape[2]
            scale = min(1.0, math.sqrt(VLM_MAX_PIXELS / (h * w)))
            nh, nw = max(round(h * scale), 28), max(round(w * scale), 28)
            if (nh, nw) != (h, w):
                t = (
                    F.interpolate(
                        t.unsqueeze(0).float(),
                        size=(nh, nw),
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .clamp(0, 1)
                )
            out.append(t.float())
        return out

    def _encode_ref_latents(self, pils, target_pixels: int, generator=None) -> List[torch.Tensor]:
        """Port of krea2.py:562-606.

        match_target_res scales each reference's AREA to the target pixel count
        with aspect preserved -- it is NOT a resize to width x height. Sides snap
        to a multiple of 16 so the latent grid is patchifiable. Bilinear, and
        deliberately WITHOUT antialias, matching the trainer.
        """
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor

        sc = self.CONFIG.resolution_divisor  # 16
        budget = target_pixels if self.match_target_res else CONTROL_IMAGE_MAX_PIXELS

        out = []
        for im in pils:
            t = to_tensor(im).unsqueeze(0).to(self.device, dtype=self.dtype)
            h, w = t.shape[2], t.shape[3]
            if self.match_target_res or h * w > budget:
                ratio = h / w
                new_h = math.sqrt(budget * ratio)
                new_w = new_h / ratio
            else:
                new_h, new_w = float(h), float(w)
            new_h = max(sc, int(round(new_h / sc)) * sc)
            new_w = max(sc, int(round(new_w / sc)) * sc)
            if (new_h, new_w) != (h, w):
                t = F.interpolate(t, size=(new_h, new_w), mode="bilinear")
            # encode_images expects [-1, 1]; drop the batch dim -> (16, h, w)
            out.append(self.pipe.encode_images(t * 2 - 1, generator=generator)[0])
        return out

    # ------------------------------------------------------------ prompting

    def _encode_prompt(self, prompt: str, vlm_images) -> torch.Tensor:
        te = _load_aitk_krea2_modules()["text_encoder"]

        feats = te.encode_krea_prompt(
            self.text_encoder,
            self.tokenizer,
            self.suffix_tokenizer,  # positional `processor` = Qwen2TokenizerFast
            prompt,
            max_length=MAX_TEXT_LENGTH,
            select_layers=te.SELECT_LAYERS,
            images=vlm_images,
            vl_processor=self.vl_processor,
            dtype=self.dtype,
        )
        # (L, 12, 2560) -> (L, 30720); predict_velocity restores the layer axis.
        return feats.reshape(feats.shape[0], -1).to(self.dtype)

    # ------------------------------------------------------------ inference

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
        num_frames: int = 1,
        fps: int = 16,
    ) -> Dict[str, Any]:
        negative_prompt = negative_prompt or ""

        refs = self._control_pils(control_image, control_images) if self.IS_EDIT else []
        if self.IS_EDIT and not refs:
            raise ValueError(
                f"{self.CONFIG.model_type.value} requires at least one control image"
            )
        vlm_images = self._prep_vlm_images(refs) if refs else None

        conditional_embeds = [self._encode_prompt(prompt, vlm_images)]

        # CFG is 0-normalized for this model: the trainer does
        # `guidance = max(0.0, gen_config.guidance_scale - 1.0)` (krea2.py:526-527)
        # before handing it to the sampler, so the request's guidance_scale means
        # the same thing here as sample.guidance_scale does in a training config.
        # Do NOT subtract this anywhere else -- doing it twice under-guides by a
        # full point.
        guidance = max(0.0, float(guidance_scale) - 1.0)

        unconditional_embeds = None
        if guidance > 0:
            # Edit mode passes the SAME references to the uncond encode
            # (toolkit/models/base_model.py); a text-only negative would change
            # the CFG direction.
            unconditional_embeds = [self._encode_prompt(negative_prompt, vlm_images)]

        ref_latents = None
        if refs:
            ref_latents = [
                self._encode_ref_latents(refs, width * height, generator=generator)
            ]

        call_kwargs = dict(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance,
            generator=generator,
            ref_latents=ref_latents,
        )
        # ComfyUI progress + interrupt; a no-op unless an observer is installed.
        self._inject_diffusers_callback_kwargs(
            call_kwargs, total_steps=num_inference_steps
        )

        return {"image": self.pipe(**call_kwargs).images[0]}

    def unload(self):
        for attr in (
            "pipe",
            "transformer",
            "vae",
            "text_encoder",
            "tokenizer",
            "suffix_tokenizer",
            "vl_processor",
        ):
            setattr(self, attr, None)
        self.lora_loaded = False
        self._current_lora_paths = []
        _flush()
        logger.info(f"{self.__class__.__name__} unloaded")


class Krea2TurboPipeline(Krea2Pipeline):
    """Krea 2 Turbo -- the step-distilled checkpoint. Separate full weights, same
    architecture.

    The ostris/krea2_turbo_training_adapter is TRAINING-ONLY. ai-toolkit merges it
    at +1.0 and then activates it at multiplier -1.0 during sampling
    (krea2.py:325, 368-381 plus base_model.py), so training previews are produced
    by plain turbo weights. It must never be loaded here.
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.KREA2_TURBO,
        base_model="krea/Krea-2-Turbo",
        resolution_divisor=16,
        default_steps=9,  # ai-toolkit ui options.ts krea2:turbo
        default_guidance_scale=1.0,  # -> internal guidance 0.0 -> CFG off
        requires_control_image=False,
        supports_negative_prompt=False,  # inert while guidance_scale <= 1
        lora_merge_method=LoraMergeMethod.CUSTOM,
        default_width=1024,
        default_height=1024,
    )
    CHECKPOINT_FILENAME = "turbo.safetensors"


class Krea2EditPipeline(Krea2Pipeline):
    """Krea 2 (raw) in-context edit -- the ai-toolkit arch krea2:o_edit.

    Reference images are consumed twice: through Qwen3-VL alongside the prompt,
    and as clean VAE latents appended to the image sequence at t=0 (the ComfyUI
    Kontext "index_timestep_zero" placement).
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.KREA2_O_EDIT,
        base_model="krea/Krea-2-Raw",
        resolution_divisor=16,
        default_steps=30,
        default_guidance_scale=4.0,
        requires_control_image=True,
        supports_negative_prompt=True,
        lora_merge_method=LoraMergeMethod.CUSTOM,
        default_width=1024,
        default_height=1024,
    )
    CHECKPOINT_FILENAME = "raw.safetensors"
    IS_EDIT = True
    # Both edit presets train with model_kwargs {edit, match_target_res, kv_cache}
    # all true. kv_cache is not an optional speedup -- it changes the attention
    # mask, so a LoRA trained with it on must be inferenced with it on.
    USE_KV_CACHE = True
    MATCH_TARGET_RES = True


class Krea2EditTurboPipeline(Krea2EditPipeline):
    """Krea 2 Turbo in-context edit -- the ai-toolkit arch krea2:o_edit_turbo."""

    CONFIG = PipelineConfig(
        model_type=ModelType.KREA2_O_EDIT_TURBO,
        base_model="krea/Krea-2-Turbo",
        resolution_divisor=16,
        # 8, not 9: upstream bumped krea2:turbo from 8 to 9 but left
        # krea2:o_edit_turbo at 8. Mirrored for preview parity -- do not "fix".
        default_steps=8,
        default_guidance_scale=1.0,
        requires_control_image=True,
        supports_negative_prompt=False,
        lora_merge_method=LoraMergeMethod.CUSTOM,
        default_width=1024,
        default_height=1024,
    )
    CHECKPOINT_FILENAME = "turbo.safetensors"
