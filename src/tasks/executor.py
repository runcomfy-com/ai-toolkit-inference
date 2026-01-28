"""Inference task executor."""

import os
import logging
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from ..schemas.task import InferenceTask, RequestStatus
from ..schemas.request import PromptItem
from ..libs.storage import InMemoryStorage
from ..libs.image_utils import (
    load_image_from_source,
    save_image,
    save_video_frames,
    save_video_with_audio,
)
from ..pipelines import get_pipeline_config, LoraMergeMethod
from ..services.pipeline_manager import PipelineManager
from ..libs.log_context import set_request_id

logger = logging.getLogger(__name__)


@dataclass
class PromptParams:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    num_frames: int
    fps: int
    control_image: Optional[Image.Image] = None
    control_images: Optional[List[Image.Image]] = None


@dataclass
class PromptResult:
    result: Dict[str, Any]
    params: PromptParams
    timing: Dict[str, float]


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        logger.info(f"[TIMING] {self.name}: {self.elapsed:.3f}s")


class InferenceExecutor:
    def __init__(
        self,
        storage: InMemoryStorage,
        pipeline_manager: PipelineManager,
        output_base_path: str = "/tmp/inference_output",
        inference_timeout: int = 3600,
    ):
        self.storage = storage
        self.pipeline_manager = pipeline_manager
        self.output_base_path = output_base_path
        self.inference_timeout = inference_timeout
        self._inference_lock = threading.Lock()
        logger.info(f"InferenceExecutor initialized with timeout={inference_timeout}s")

    def execute(self, task: InferenceTask):
        # Set request_id in logging context for background task
        set_request_id(task.id)
        logger.info(f"Waiting for inference lock...")

        with self._inference_lock:
            logger.info(f"Acquired inference lock (timeout={self.inference_timeout}s)")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._execute_locked, task)
                try:
                    future.result(timeout=self.inference_timeout)
                except FuturesTimeoutError:
                    logger.error(f"Task timed out after {self.inference_timeout}s")
                    task.mark_as_failed(
                        error=f"Inference timed out after {self.inference_timeout} seconds",
                        details={"timeout_seconds": self.inference_timeout},
                    )
                    self.storage.update(task)

    def _execute_locked(self, task: InferenceTask):
        # Ensure request_id is set in this thread's context
        set_request_id(task.id)
        timings = {}
        pipeline = None

        try:
            logger.info("Starting inference task")
            task.mark_as_processing()
            self.storage.update(task)

            pipeline_config = get_pipeline_config(task.model)
            if not pipeline_config:
                raise ValueError(f"Unknown model: {task.model}")

            prompts = [PromptItem(**p) for p in task.inputs.get("prompts", [])]
            self._replace_trigger_word(prompts, task.inputs.get("trigger_word"))

            pipeline_defaults = self._get_pipeline_defaults(pipeline_config)
            base_lora_scale = self._get_base_lora_scale(task.inputs.get("lora_scales"), pipeline_defaults)
            initial_lora_scale = self._get_initial_scale(prompts, base_lora_scale, pipeline_defaults)
            needs_per_prompt_scale = self._has_multiple_scales(prompts, base_lora_scale, pipeline_defaults)

            lora_paths = task.lora_paths or []
            pipeline_timings = {}
            with Timer("get_pipeline") as t:
                pipeline = self.pipeline_manager.get_pipeline(
                    model=task.model,
                    lora_paths=lora_paths,
                    lora_scale=initial_lora_scale,
                    hf_token=task.inputs.get("hf_token"),
                    timings=pipeline_timings,
                )
            timings["get_pipeline"] = t.elapsed
            timings["pipeline_details"] = pipeline_timings

            with Timer("inference_total") as t:
                outputs = self._run_inference(
                    task=task,
                    pipeline=pipeline,
                    prompts=prompts,
                    pipeline_defaults=pipeline_defaults,
                    pipeline_config=pipeline_config,
                    timings=timings,
                    lora_paths=lora_paths,
                    needs_per_prompt_scale=needs_per_prompt_scale,
                    initial_lora_scale=initial_lora_scale,
                    base_lora_scale=base_lora_scale,
                )
            timings["inference_total"] = t.elapsed

            total_time = (datetime.now(timezone.utc) - task.started_at).total_seconds() if task.started_at else None
            task.mark_as_succeeded(outputs, {"model": task.model, "inference_time": total_time, "timings": timings})
            self.storage.update(task)
            logger.info(f"Completed inference task in {total_time:.1f}s")

        except Exception as e:
            logger.error(f"Failed inference task: {e}")
            logger.error(traceback.format_exc())
            task.mark_as_failed(error=str(e), details={"traceback": traceback.format_exc()})
            self.storage.update(task)

        # Note: Pipeline is now cached by PipelineManager, no need to unload after each request
        # The pipeline will be reused for subsequent requests with the same model
        # To force unload, call pipeline_manager.force_unload()

    def _run_inference(
        self,
        task: InferenceTask,
        pipeline: Any,
        prompts: List[PromptItem],
        pipeline_defaults: Dict[str, Any],
        pipeline_config: Any,
        timings: Dict[str, Any],
        lora_paths: List[str] = None,
        needs_per_prompt_scale: bool = False,
        initial_lora_scale: float | Dict[str, float] = 1.0,
        base_lora_scale: float | Dict[str, float] = 1.0,
    ) -> Dict[str, Any]:
        os.makedirs(self.output_base_path, exist_ok=True)
        file_prefix = f"{task.id}_"

        # CUSTOM merge method cannot change scale dynamically
        if needs_per_prompt_scale and pipeline_config.lora_merge_method == LoraMergeMethod.CUSTOM:
            logger.info(f"Pipeline {task.model} requires grouped processing for per-prompt lora_scale")
            return self._run_inference_with_reload(
                task,
                pipeline,
                prompts,
                pipeline_defaults,
                pipeline_config,
                timings,
                lora_paths or [],
                initial_lora_scale,
                base_lora_scale,
                file_prefix,
            )

        outputs = {"images": [], "videos": []}
        base_seed = pipeline_defaults.get("seed", 42)
        divisor = pipeline_defaults.get("resolution_divisor", 8)
        prompt_timings = []

        for i, prompt_item in enumerate(prompts):
            # SET_ADAPTERS and FUSE modes support dynamic scale (FUSE via unfuse/fuse cycle)
            if needs_per_prompt_scale and pipeline_config.lora_merge_method != LoraMergeMethod.CUSTOM:
                scale = self._get_prompt_scale(prompt_item, base_lora_scale, pipeline_defaults)
                pipeline.set_lora_scale(scale)
                logger.info(f"Set lora_scale: {scale}")

            prompt_result = self._process_single_prompt(
                pipeline,
                prompt_item,
                pipeline_config,
                pipeline_defaults,
                i,
                len(prompts),
                base_seed,
                divisor,
            )
            self._save_result(prompt_result, pipeline_config, file_prefix, i, outputs)
            prompt_timings.append(prompt_result.timing)

        timings["prompts"] = prompt_timings
        return outputs

    def _run_inference_with_reload(
        self,
        task: InferenceTask,
        pipeline: Any,
        prompts: List[PromptItem],
        pipeline_defaults: Dict[str, Any],
        pipeline_config: Any,
        timings: Dict[str, Any],
        lora_paths: List[str],
        initial_lora_scale: float | Dict[str, float],
        base_lora_scale: float | Dict[str, float],
        file_prefix: str,
    ) -> Dict[str, Any]:
        groups = self._group_prompts_by_scale(prompts, base_lora_scale, pipeline_defaults)
        group_scales = [group["scale"] for group in groups.values()]
        logger.info(f"Grouped {len(prompts)} prompts into {len(groups)} scale groups: {group_scales}")

        results_by_idx: Dict[int, PromptResult] = {}
        current_pipeline = pipeline
        current_scale = initial_lora_scale
        reload_count = 0

        base_seed = pipeline_defaults.get("seed", 42)
        divisor = pipeline_defaults.get("resolution_divisor", 8)

        try:
            for group in groups.values():
                scale = group["scale"]
                indexed_prompts = group["items"]
                if scale != current_scale:
                    reload_count += 1
                    logger.info(f"Reloading pipeline: {current_scale} -> {scale} (reload #{reload_count})")

                    with Timer(f"reload_pipeline_{reload_count}") as t:
                        self.pipeline_manager.unload_pipeline(current_pipeline)
                        current_pipeline = self.pipeline_manager.get_pipeline(
                            model=task.model,
                            lora_paths=lora_paths,
                            lora_scale=scale,
                            hf_token=task.inputs.get("hf_token"),
                        )
                        current_scale = scale
                    timings[f"reload_pipeline_{reload_count}"] = t.elapsed

                for original_idx, prompt_item in indexed_prompts:
                    prompt_result = self._process_single_prompt(
                        current_pipeline,
                        prompt_item,
                        pipeline_config,
                        pipeline_defaults,
                        original_idx,
                        len(prompts),
                        base_seed,
                        divisor,
                        extra_log=f" (scale={scale})",
                    )
                    results_by_idx[original_idx] = prompt_result
        finally:
            pass

        timings["pipeline_reloads"] = reload_count

        outputs = {"images": [], "videos": []}
        prompt_timings = []

        for i in range(len(prompts)):
            prompt_result = results_by_idx[i]
            self._save_result(prompt_result, pipeline_config, file_prefix, i, outputs)
            prompt_timings.append(prompt_result.timing)

        timings["prompts"] = prompt_timings
        return outputs

    def _process_single_prompt(
        self,
        pipeline: Any,
        prompt_item: PromptItem,
        pipeline_config: Any,
        pipeline_defaults: Dict[str, Any],
        prompt_idx: int,
        total_prompts: int,
        base_seed: int,
        divisor: int,
        extra_log: str = "",
    ) -> PromptResult:
        timing = {}

        params = self._prepare_prompt_params(prompt_item, pipeline_defaults, divisor, base_seed, prompt_idx)
        params.control_image, params.control_images = self._load_control_images(
            prompt_item, pipeline_config, prompt_idx, timing
        )
        self._log_generate_params(params, prompt_idx, total_prompts, extra_log)

        with Timer(f"prompt_{prompt_idx}_generate") as t:
            result = pipeline.generate(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                width=params.width,
                height=params.height,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                seed=params.seed,
                control_image=params.control_image,
                control_images=params.control_images,
                num_frames=params.num_frames,
                fps=params.fps,
            )
        timing["generate"] = t.elapsed

        return PromptResult(result=result, params=params, timing=timing)

    def _save_result(
        self,
        prompt_result: PromptResult,
        pipeline_config: Any,
        file_prefix: str,
        prompt_idx: int,
        outputs: Dict[str, List],
    ) -> None:
        if pipeline_config.is_video_model:
            self._save_video_result(prompt_result, file_prefix, prompt_idx, outputs)
        else:
            self._save_image_result(prompt_result, file_prefix, prompt_idx, outputs)

    def _save_video_result(
        self,
        prompt_result: PromptResult,
        file_prefix: str,
        prompt_idx: int,
        outputs: Dict[str, List],
    ) -> None:
        result = prompt_result.result
        params = prompt_result.params
        timing = prompt_result.timing

        video_tensor = result.get("video_tensor")
        frames = result.get("frames")
        audio = result.get("audio")
        audio_sample_rate = result.get("audio_sample_rate", 24000)

        num_frames_out = video_tensor.shape[0] if video_tensor is not None else (len(frames) if frames else 0)

        with Timer(f"prompt_{prompt_idx}_save_video") as t:
            if audio is not None or video_tensor is not None:
                output_path = os.path.join(self.output_base_path, f"{file_prefix}output_{prompt_idx}.mp4")
                save_video_with_audio(
                    output_path=output_path,
                    fps=params.fps,
                    audio=audio,
                    audio_sample_rate=audio_sample_rate,
                    video_tensor=video_tensor,
                    frames=frames,
                )
                output_format = "mp4"
            else:
                output_path = os.path.join(self.output_base_path, f"{file_prefix}output_{prompt_idx}.webp")
                save_video_frames(frames, output_path, fps=params.fps, format="WEBP")
                output_format = "webp"
        timing["save"] = t.elapsed

        outputs["videos"].append(
            {
                "format": output_format,
                "width": params.width,
                "height": params.height,
                "num_frames": num_frames_out,
                "fps": params.fps,
                "file_path": output_path,
                "seed": result.get("seed", params.seed),
            }
        )

    def _save_image_result(
        self,
        prompt_result: PromptResult,
        file_prefix: str,
        prompt_idx: int,
        outputs: Dict[str, List],
    ) -> None:
        result = prompt_result.result
        params = prompt_result.params
        timing = prompt_result.timing

        image = result["image"]

        with Timer(f"prompt_{prompt_idx}_save_image") as t:
            output_path = os.path.join(self.output_base_path, f"{file_prefix}output_{prompt_idx}.jpg")
            save_image(image, output_path, format="JPEG", quality=95)
        timing["save"] = t.elapsed

        outputs["images"].append(
            {
                "format": "jpeg",
                "width": image.width,
                "height": image.height,
                "file_path": output_path,
                "seed": result.get("seed", params.seed),
            }
        )

    def _prepare_prompt_params(
        self,
        prompt_item: PromptItem,
        pipeline_defaults: Dict[str, Any],
        divisor: int,
        base_seed: int,
        prompt_idx: int,
    ) -> PromptParams:
        width = prompt_item.width if prompt_item.width is not None else pipeline_defaults["width"]
        height = prompt_item.height if prompt_item.height is not None else pipeline_defaults["height"]

        return PromptParams(
            prompt=prompt_item.prompt,
            negative_prompt=prompt_item.neg if prompt_item.neg is not None else pipeline_defaults.get("neg", ""),
            width=(width // divisor) * divisor,
            height=(height // divisor) * divisor,
            num_inference_steps=prompt_item.sample_steps
            if prompt_item.sample_steps is not None
            else pipeline_defaults["sample_steps"],
            guidance_scale=prompt_item.guidance_scale
            if prompt_item.guidance_scale is not None
            else pipeline_defaults["guidance_scale"],
            seed=prompt_item.seed
            if prompt_item.seed is not None
            else (base_seed + prompt_idx if base_seed >= 0 else -1),
            num_frames=prompt_item.num_frames
            if prompt_item.num_frames is not None
            else pipeline_defaults.get("num_frames", 1),
            fps=prompt_item.fps if prompt_item.fps is not None else pipeline_defaults.get("fps", 16),
        )

    def _load_control_images(
        self,
        prompt_item: PromptItem,
        pipeline_config: Any,
        prompt_idx: int,
        timing: Dict[str, float],
    ) -> Tuple[Optional[Image.Image], Optional[List[Image.Image]]]:
        ctrl_img_sources = prompt_item.get_control_images()
        should_load = pipeline_config.requires_control_image or len(ctrl_img_sources) > 0

        if not should_load or not ctrl_img_sources:
            return None, None

        control_image = None
        control_images = None

        with Timer(f"prompt_{prompt_idx}_load_control_images") as t:
            if len(ctrl_img_sources) > 1:
                control_images = []
                for src in ctrl_img_sources:
                    img = load_image_from_source(
                        base64_data=src if not src.startswith("http") else None,
                        url=src if src.startswith("http") else None,
                    )
                    if img:
                        control_images.append(img)
                logger.info(f"Loaded {len(control_images)} control images")
            else:
                src = ctrl_img_sources[0]
                control_image = load_image_from_source(
                    base64_data=src if not src.startswith("http") else None,
                    url=src if src.startswith("http") else None,
                )
        timing["load_control_images"] = t.elapsed

        return control_image, control_images

    def _get_pipeline_defaults(self, pipeline_config: Any) -> Dict[str, Any]:
        return {
            "width": pipeline_config.default_width,
            "height": pipeline_config.default_height,
            "sample_steps": pipeline_config.default_steps,
            "guidance_scale": pipeline_config.default_guidance_scale,
            "neg": pipeline_config.default_neg,
            "seed": pipeline_config.default_seed,
            "network_multiplier": pipeline_config.default_network_multiplier,
            "num_frames": pipeline_config.default_num_frames,
            "fps": pipeline_config.default_fps,
            "resolution_divisor": pipeline_config.resolution_divisor,
        }

    def _replace_trigger_word(self, prompts: List[PromptItem], trigger_word: Optional[str]) -> None:
        for prompt_item in prompts:
            local_trigger = prompt_item.trigger_word or trigger_word
            if not local_trigger:
                continue
            if "[trigger]" in prompt_item.prompt:
                prompt_item.prompt = prompt_item.prompt.replace("[trigger]", local_trigger)
                logger.info(f"Replaced [trigger] with '{local_trigger}'")

    def _get_base_lora_scale(
        self,
        lora_scales: float | Dict[str, float] | None,
        pipeline_defaults: Dict[str, Any],
    ) -> float | Dict[str, float]:
        default_scale = pipeline_defaults["network_multiplier"]
        if isinstance(lora_scales, dict):
            return {key: default_scale if value is None else value for key, value in lora_scales.items()}
        if isinstance(lora_scales, (int, float)):
            return float(lora_scales)
        return default_scale

    def _get_initial_scale(
        self,
        prompts: List[PromptItem],
        base_lora_scale: float | Dict[str, float],
        pipeline_defaults: Dict[str, Any],
    ) -> float | Dict[str, float]:
        return base_lora_scale

    def _scale_key(self, scale: float | Dict[str, float]) -> float | tuple:
        if isinstance(scale, dict):
            return tuple(sorted(scale.items()))
        return scale

    def _has_multiple_scales(
        self,
        prompts: List[PromptItem],
        base_lora_scale: float | Dict[str, float],
        pipeline_defaults: Dict[str, Any],
    ) -> bool:
        return False

    def _get_prompt_scale(
        self,
        prompt_item: PromptItem,
        base_lora_scale: float | Dict[str, float],
        pipeline_defaults: Dict[str, Any],
    ) -> float | Dict[str, float]:
        return base_lora_scale

    def _group_prompts_by_scale(
        self,
        prompts: List[PromptItem],
        base_lora_scale: float | Dict[str, float],
        pipeline_defaults: Dict[str, Any],
    ) -> OrderedDict[float | tuple, Dict[str, Any]]:
        return OrderedDict(
            {
                self._scale_key(base_lora_scale): {
                    "scale": base_lora_scale,
                    "items": list(enumerate(prompts)),
                }
            }
        )

    def _log_generate_params(
        self,
        params: PromptParams,
        prompt_idx: int,
        total_prompts: int,
        extra_info: str = "",
    ) -> None:
        logger.info(f"=== Generate [{prompt_idx + 1}/{total_prompts}]{extra_info} ===")
        logger.info(f"  prompt: {params.prompt}")
        logger.info(
            f"  size: {params.width}x{params.height}, steps: {params.num_inference_steps}, guidance: {params.guidance_scale}"
        )
        logger.info(f"  seed: {params.seed}, frames: {params.num_frames}, fps: {params.fps}")
        if params.control_image:
            logger.info(f"  control_image: {params.control_image.size}")
        if params.control_images:
            logger.info(f"  control_images: {len(params.control_images)} images")
