"""
Anima pipeline (text-to-image).

Anima is a Cosmos-DiT flow-matching text-to-image model:
  - denoiser:      CosmosTransformer3DModel                    (diffusers)
  - text encoder:  Qwen3Model + a learned AnimaTextConditioner (diffusers)
                   that fuses the Qwen states with T5 token ids
  - autoencoder:   the Qwen-Image VAE (AutoencoderKLQwenImage, f8)
  - base weights:  circlestone-labs/Anima-Base-v1.0-Diffusers

Unlike Krea 2, Anima HAS a real diffusers pipeline: the *modular* pipeline
`AnimaAutoBlocks` / `AnimaModularPipeline`, added to diffusers in
huggingface/diffusers#13732 (commit c943837, 2026-05-29). So this wrapper does
NOT hand-roll the sampler. Instead it reproduces the ai-toolkit trainer's
generation path *exactly* -- see
extensions_built_in/diffusion_models/anima/anima.py in ostris/ai-toolkit:

  * load: AnimaAutoBlocks().init_pipeline(repo).load_components(torch_dtype)   (anima.py:257-262)
  * scheduler: replace with CustomFlowMatchEulerDiscreteScheduler(**cfg)        (anima.py:243-244, 263)
  * generate: build the EMBEDS-path pipeline [AnimaCoreDenoiseStep,
              AnimaDecodeStep] and drive it with pre-computed Qwen/T5 embeds     (anima.py:213-215, 334-356)
  * text encode: our own _get_qwen_prompt_embeds / _get_t5_prompt_ids           (anima.py:505-568)

WHY the embeds path and not the one-shot `AnimaAutoBlocks` (prompt-string) path:
the trainer's `_get_qwen_prompt_embeds` applies an EMPTY-PROMPT MASK FIX --
for a prompt whose attention mask is all zeros (the empty negative prompt under
CFG) it forces the first mask position to 1 so the text conditioner sees one
live token instead of an all-zero sequence (anima.py:528-531, 539). Diffusers'
own AnimaTextEncoderStep does NOT do this (modular_pipelines/anima/encoders.py:
_get_qwen_prompt_embeds zeros everything for an empty prompt), so routing the
uncond branch through AnimaAutoBlocks would change the CFG direction and drift
from the training preview. This repo's product is training-preview parity, so we
mirror the trainer.

CFG: Anima uses RAW guidance -- the trainer sets
`pipeline.guider.guidance_scale = gen_config.guidance_scale` with NO -1
(anima.py:444). This is the OPPOSITE of Krea 2 (which 0-normalizes with
`max(0, g-1)`). Do not subtract anything here.

IMPORT SAFETY: every diffusers-Anima / ai-toolkit import lives inside a method.
src/api/v1/inference.py imports PIPELINE_REGISTRY at module scope and
src/pipelines/__init__.py builds it with no try/except, so a top-level import of
`AnimaAutoBlocks` would break FastAPI startup for every model whenever the
container's diffusers predates c943837. The registry is built from CONFIG only,
which needs none of these imports.

NOTE (diffusers pin): Anima needs diffusers >= c943837. The inference image
historically pinned 072d15ee (LTX-2.3, 2026-03-19), which is 203 commits older
and has no Anima module. requirements-inference.txt is bumped to c943837 (the
ai-toolkit training pin, already proven to host flux2/krea2/qwen_image/ltx2/
anima together) alongside this file. That bump is a shared dependency for every
pipeline and MUST be regression-tested on a GPU before release.
"""

import gc
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from .base import BasePipeline, LoraMergeMethod, PipelineConfig
from ..schemas.models import ModelType

logger = logging.getLogger(__name__)

# anima.py:28-42 -- the trainer's hardcoded scheduler config. The trainer
# REPLACES the pipeline's repo scheduler with this exact config
# (get_train_scheduler, anima.py:243-244), so we do too.
SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_dynamic_shifting": False,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# ai-toolkit ui/src/app/jobs/new/options.ts:84-87 -- the neg prompt the Anima
# preset fills in by default. Applied when a request sends an empty neg, so an
# unspecified negative prompt matches the training-preview default rather than
# an empty string. (Only relevant while guidance_scale > 1, i.e. CFG is on.)
ANIMA_DEFAULT_NEG = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, "
    "jpeg artifacts, sepia, signature, artist name"
)

# anima.py:240 -- Qwen tokenizer / T5 tokenizer truncation length.
MAX_SEQUENCE_LENGTH = 512

# anima.py:246 get_bucket_divisibility() returns 16*2 = 32. The modular pipeline
# only *requires* divisibility by vae_scale_factor*2 = 16 (AnimaPrepareLatentsStep
# .check_inputs), but the trainer rounds preview W/H to 32 (generate_single_image,
# anima.py:438-440), so we snap to 32 for parity. 32 is a multiple of 16, so the
# pipeline's own check is always satisfied.
RESOLUTION_DIVISOR = 32


def _flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class AnimaPipeline(BasePipeline):
    """Anima text-to-image."""

    CONFIG = PipelineConfig(
        model_type=ModelType.ANIMA,
        base_model="circlestone-labs/Anima-Base-v1.0-Diffusers",
        resolution_divisor=RESOLUTION_DIVISOR,
        default_steps=30,  # ai-toolkit ui/src/helpers/defaultSamples.ts:48
        default_guidance_scale=4.0,  # defaultSamples.ts:47 -- RAW (no -1)
        requires_control_image=False,
        supports_negative_prompt=True,
        lora_merge_method=LoraMergeMethod.CUSTOM,  # fused in _load_pipeline
        default_width=1024,
        default_height=1024,
        default_neg=ANIMA_DEFAULT_NEG,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.text_conditioner = None
        self.vae = None
        self.text_encoder = None  # Qwen3Model
        self.tokenizer = None  # Qwen2Tokenizer
        self.t5_tokenizer = None  # T5Tokenizer
        self._lora_path = None
        self._lora_scale = 1.0

    # ------------------------------------------------------------------ load

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Override BasePipeline.load (the krea2.py / flux2.py shape).

        Skips _enable_xformers / _apply_offload_mode: `self.pipe` is a modular
        pipeline, components are placed manually below, and the LoRA is fused
        during _load_pipeline (CUSTOM merge method), so those base steps would
        be no-ops at best.
        """
        if lora_paths:
            first = lora_paths[0]
            self._lora_path = list(first.values())[0] if isinstance(first, dict) else first
        self._lora_scale = float(lora_scale)
        logger.info(
            f"Loading {self.__class__.__name__} lora={self._lora_path} scale={self._lora_scale}"
        )
        if self.offload_mode != "none":
            logger.warning(
                "offload_mode=%s is ignored by Anima (components are placed manually)",
                self.offload_mode,
            )
        self._load_pipeline()

    def _build_scheduler(self):
        """The trainer's get_train_scheduler() (anima.py:242-244).

        CustomFlowMatchEulerDiscreteScheduler overrides only __init__ (it adds
        training-only loss-weighting tables); it does NOT override set_timesteps
        or step, so for the inference path it is behaviourally identical to
        diffusers' FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG). We import
        the trainer's class anyway so the schedule cannot silently drift if
        ostris changes it later. ai-toolkit is a hard requirement of this image
        (see krea2.py), so this import is safe.
        """
        from toolkit.samplers.custom_flowmatch_sampler import (
            CustomFlowMatchEulerDiscreteScheduler,
        )

        return CustomFlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG)

    def _load_pipeline(self):
        # Heavy / version-sensitive imports are method-scoped (see module docstring).
        from diffusers import AnimaAutoBlocks
        from diffusers.modular_pipelines import SequentialPipelineBlocks
        from diffusers.modular_pipelines.anima.modular_blocks_anima import (
            AnimaCoreDenoiseStep,
            AnimaDecodeStep,
        )

        _flush()

        # -- 1. load the full model via the AUTO blocks -----------------------
        # AnimaAutoBlocks bundles [text_encoder, denoise, decode], so
        # load_components() pulls every component we need: text_encoder (Qwen3),
        # tokenizer (Qwen2), t5_tokenizer (T5), text_conditioner, transformer
        # (Cosmos), vae (Qwen-Image), scheduler, guider, image_processor.
        # anima.py:257-262.
        logger.info(f"[1/4] Loading Anima components ({self.CONFIG.base_model})")
        loader = AnimaAutoBlocks().init_pipeline(self.CONFIG.base_model)
        loader.load_components(torch_dtype=self.dtype)

        # -- 2. replace the scheduler (anima.py:263) --------------------------
        loader.update_components(scheduler=self._build_scheduler())

        # -- 3. fuse the user LoRA (anima.py:668-673 conversion, run via the
        #        AnimaLoraLoaderMixin that AnimaModularPipeline inherits) -------
        # lora_state_dict() internally calls _convert_non_diffusers_anima_lora_to_
        # diffusers (comfy `diffusion_model.*` -> diffusers `transformer.*` /
        # `text_conditioner.*`) and load_lora_weights() routes each to the right
        # component. We fuse at the requested scale (CUSTOM merge: no dynamic
        # scale change, a new scale requires a reload -- same as Krea 2 / Flux2).
        if self._lora_path and os.path.exists(self._lora_path):
            logger.info(f"[2/4] Fusing LoRA {self._lora_path} scale={self._lora_scale}")
            self._fuse_lora(loader, self._lora_path, self._lora_scale)
            self.lora_loaded = True
        elif self._lora_path:
            logger.error(f"LoRA file not found: {self._lora_path}")
        else:
            logger.warning("No LoRA path provided; running base weights only")

        # -- 4. place components on the device --------------------------------
        logger.info(f"[3/4] Moving components to {self.device}")
        loader.to(self.device)

        self.transformer = loader.transformer
        self.text_conditioner = loader.text_conditioner
        self.vae = loader.vae
        self.text_encoder = loader.text_encoder
        self.tokenizer = loader.tokenizer
        self.t5_tokenizer = loader.t5_tokenizer

        # -- 5. build the EMBEDS-path generation pipeline (anima.py:213-215,
        #        334-343) -- text encoding is done by us, so the pipeline is
        #        [denoise, decode] only and consumes pre-computed embeds. ------
        logger.info("[4/4] Building Anima embeds->image pipeline")

        class _AnimaEmbedsToImageBlocks(SequentialPipelineBlocks):
            # Local copy of ai-toolkit's AnimaEmbedsToImageBlocks (anima.py:213).
            model_name = "anima"
            block_classes = [AnimaCoreDenoiseStep, AnimaDecodeStep]
            block_names = ["denoise", "decode"]

        gen = _AnimaEmbedsToImageBlocks().init_pipeline()
        gen.update_components(
            scheduler=self._build_scheduler(),
            transformer=self.transformer,
            text_conditioner=self.text_conditioner,
            vae=self.vae,
        )
        gen = gen.to(self.device)
        self._disable_progress_bars(gen)
        self.pipe = gen

        _flush()
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB allocated")
        logger.info(f"{self.__class__.__name__} loaded")

    @staticmethod
    def _disable_progress_bars(pipeline):
        """anima.py:345-355. ModularPipeline.set_progress_bar_config only walks
        one level of sub_blocks, but the tqdm bar lives two levels deep in the
        denoise loop. Recurse over `_blocks` (the public `.blocks` returns a
        fresh copy each access)."""

        def _walk(blocks):
            sub = getattr(blocks, "sub_blocks", None)
            if not sub:
                return
            for sub_block in sub.values():
                if hasattr(sub_block, "set_progress_bar_config"):
                    sub_block.set_progress_bar_config(disable=True)
                _walk(sub_block)

        try:
            _walk(pipeline._blocks)
        except Exception as e:  # pragma: no cover - cosmetic only
            logger.debug(f"Could not disable Anima progress bars: {e}")

    # ------------------------------------------------------------------ LoRA

    def _fuse_lora(self, pipeline, lora_path: str, lora_scale: float):
        """Load + fuse an ai-toolkit-trained Anima LoRA via AnimaLoraLoaderMixin.

        On-disk keys are comfy format (`diffusion_model.*`); the mixin's
        lora_state_dict() converts them and load_lora_weights() routes to the
        transformer (and text_conditioner, if the adapter trained it -- only
        when the training config set model_kwargs.train_text_conditioner).
        """
        from safetensors.torch import load_file

        state_dict = load_file(lora_path, device="cpu")
        pipeline.load_lora_weights(state_dict, adapter_name="lora")
        pipeline.fuse_lora(adapter_names=["lora"], lora_scale=lora_scale)
        # Keep the adapter layers loaded (do not unload): unload_lora_weights()
        # would drop the PEFT layers and make a later unfuse impossible. Memory
        # cost is negligible since the delta is already baked into base weights.
        self._lora_fused = True
        self._num_loras_fused = 1

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0, hotswap: bool = False):
        """LoRA is fused during _load_pipeline (the krea2.py / flux2.py shape)."""
        if not self.lora_loaded:
            logger.warning("_load_lora called but Anima fuses during _load_pipeline()")

    # ------------------------------------------------------------ prompting

    def _get_qwen_prompt_embeds(self, prompt: List[str]):
        """Port of anima.py:505-541, including the empty-prompt mask fix."""
        device = self.device
        text_inputs = self.tokenizer(
            prompt,
            padding="longest",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)

        if text_input_ids.shape[1] == 0:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 151643  # anima.py:519
            text_input_ids = torch.full(
                (len(prompt), 1), pad_token_id, dtype=torch.long, device=device
            )
            prompt_attention_mask = torch.zeros_like(text_input_ids)

        # Empty-prompt fix (anima.py:528-531): give an all-zero mask one live
        # token so the text conditioner does not receive a degenerate sequence.
        conditioner_attention_mask = prompt_attention_mask.clone()
        empty_prompt_mask = conditioner_attention_mask.sum(dim=1) == 0
        if empty_prompt_mask.any():
            conditioner_attention_mask[empty_prompt_mask, 0] = 1

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)
        prompt_embeds = prompt_embeds * conditioner_attention_mask.to(prompt_embeds).unsqueeze(-1)

        return prompt_embeds, conditioner_attention_mask

    def _get_t5_prompt_ids(self, prompt: List[str]):
        """Port of anima.py:543-551."""
        device = self.device
        text_inputs = self.t5_tokenizer(
            prompt,
            padding="longest",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Mirror anima.py:553-568 get_prompt_embeds -> the four embed tensors."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = ["" if p is None else p for p in prompt]  # _normalize_prompts, anima.py:500-503
        qwen_prompt_embeds, qwen_attention_mask = self._get_qwen_prompt_embeds(prompt)
        t5_input_ids, t5_attention_mask = self._get_t5_prompt_ids(prompt)
        return {
            "qwen_prompt_embeds": qwen_prompt_embeds,
            "qwen_attention_mask": qwen_attention_mask,
            "t5_input_ids": t5_input_ids,
            "t5_attention_mask": t5_attention_mask,
        }

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
        control_image: Optional[Any] = None,
        control_images: Optional[list] = None,
        num_frames: int = 1,
        fps: int = 16,
        **kwargs,
    ) -> Dict[str, Any]:
        # Unspecified neg -> the Anima preset default, matching training previews.
        negative_prompt = negative_prompt if negative_prompt else self.CONFIG.default_neg

        # Snap to 32 exactly like generate_single_image (anima.py:438-440). The
        # base generate() already floors to resolution_divisor, so this is a
        # belt-and-suspenders guard.
        sc = RESOLUTION_DIVISOR
        width = int(width // sc * sc)
        height = int(height // sc * sc)

        cond = self._encode_prompt(prompt)
        uncond = self._encode_prompt(negative_prompt)

        # RAW guidance -- anima.py:444. No -1. The guider decides CFG on/off from
        # guidance_scale (>1 => uncond branch used).
        self.pipe.guider.guidance_scale = float(guidance_scale)

        call_kwargs = dict(
            qwen_prompt_embeds=cond["qwen_prompt_embeds"],
            qwen_attention_mask=cond["qwen_attention_mask"],
            t5_input_ids=cond["t5_input_ids"],
            t5_attention_mask=cond["t5_attention_mask"],
            negative_qwen_prompt_embeds=uncond["qwen_prompt_embeds"],
            negative_qwen_attention_mask=uncond["qwen_attention_mask"],
            negative_t5_input_ids=uncond["t5_input_ids"],
            negative_t5_attention_mask=uncond["t5_attention_mask"],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            latents=None,  # pipeline samples noise from `generator` (AnimaPrepareLatentsStep)
            generator=generator,
            output="images",
        )

        # ComfyUI progress + interrupt. The modular denoise loop exposes no
        # callback_on_step_end, so we hook the scheduler's step() for the
        # duration of the call (no-op when no observer is installed).
        with self._step_observer_hook(num_inference_steps):
            images = self.pipe(**call_kwargs)

        image = images[0] if isinstance(images, (list, tuple)) else images
        return {"image": image}

    def _step_observer_hook(self, total_steps: int):
        """Context manager: fire the base pipeline_step_observer once per
        scheduler.step(), so ComfyUI gets progress + can interrupt (by raising).

        The modular pipeline has no callback parameter, so this wraps the
        scheduler's step in place and restores it on exit. Defensive: any failure
        to install the hook degrades to no per-step callback, never an error.
        """
        from contextlib import contextmanager

        from .base import _PIPELINE_STEP_OBSERVER

        @contextmanager
        def _hook():
            observer = _PIPELINE_STEP_OBSERVER.get()
            scheduler = getattr(self.pipe, "scheduler", None)
            if observer is None or scheduler is None or not hasattr(scheduler, "step"):
                yield
                return

            original_step = scheduler.step
            state = {"i": 0}

            def wrapped_step(*args, **kwargs):
                out = original_step(*args, **kwargs)
                i = state["i"]
                state["i"] += 1
                timestep = args[1] if len(args) >= 2 else kwargs.get("timestep")
                observer(i, total_steps, timestep)  # may raise to interrupt
                return out

            try:
                scheduler.step = wrapped_step
                yield
            finally:
                scheduler.step = original_step

        return _hook()

    def unload(self):
        for attr in (
            "pipe",
            "transformer",
            "text_conditioner",
            "vae",
            "text_encoder",
            "tokenizer",
            "t5_tokenizer",
        ):
            setattr(self, attr, None)
        self.lora_loaded = False
        self._current_lora_paths = []
        self._lora_fused = False
        self._num_loras_fused = 0
        _flush()
        logger.info(f"{self.__class__.__name__} unloaded")
