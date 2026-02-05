"""
CLI for pre-downloading model assets.

Examples:
  python -m src.cli.download_models --model-type flux2
  python -m src.cli.download_models --model-type flux2,sdxl
  python -m src.cli.download_models
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..config import settings
from ..pipelines import get_pipeline_config
from ..schemas.models import ModelType
from ..services.download_config import get_download_config


@dataclass
class DownloadTask:
    repo_id: str
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None


def _normalize_patterns(patterns: Optional[List[str]]) -> Optional[List[str]]:
    if not patterns:
        return None
    return list(patterns)


def _merge_allow_patterns(a: Optional[List[str]], b: Optional[List[str]]) -> Optional[List[str]]:
    if a is None or b is None:
        return None
    merged = sorted(set(a) | set(b))
    return merged or None


def _merge_ignore_patterns(a: Optional[List[str]], b: Optional[List[str]]) -> Optional[List[str]]:
    if a is None or b is None:
        return None
    merged = sorted(set(a) & set(b))
    return merged or None


def _merge_download_tasks(tasks: Iterable[DownloadTask]) -> List[DownloadTask]:
    merged: dict[str, DownloadTask] = {}
    for task in tasks:
        repo_id = task.repo_id
        allow_patterns = _normalize_patterns(task.allow_patterns)
        ignore_patterns = _normalize_patterns(task.ignore_patterns)
        if repo_id not in merged:
            merged[repo_id] = DownloadTask(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            continue

        existing = merged[repo_id]
        existing.allow_patterns = _merge_allow_patterns(existing.allow_patterns, allow_patterns)
        existing.ignore_patterns = _merge_ignore_patterns(existing.ignore_patterns, ignore_patterns)

    return [merged[key] for key in sorted(merged.keys())]


def _collect_pipeline_configs() -> list:
    logger = logging.getLogger(__name__)
    configs = []
    for model_type in ModelType:
        try:
            config = get_pipeline_config(model_type.value)
        except Exception as e:
            logger.warning("Skipping model type %s due to import error: %s", model_type.value, e)
            continue
        if config:
            configs.append(config)
    configs.sort(key=lambda cfg: cfg.model_type.value)
    return configs


def _build_download_tasks(pipeline_configs: Iterable) -> List[DownloadTask]:
    tasks: List[DownloadTask] = []
    for cfg in pipeline_configs:
        download_cfg = get_download_config(cfg.model_type)
        tasks.append(
            DownloadTask(
                repo_id=cfg.base_model,
                allow_patterns=download_cfg.allow_patterns,
                ignore_patterns=download_cfg.ignore_patterns,
            )
        )
        if cfg.transformer_model:
            tasks.append(DownloadTask(repo_id=cfg.transformer_model))
        for extra in download_cfg.extras:
            tasks.append(
                DownloadTask(
                    repo_id=extra.repo_id,
                    allow_patterns=extra.allow_patterns,
                    ignore_patterns=extra.ignore_patterns,
                )
            )
    return _merge_download_tasks(tasks)


def _parse_model_types(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    model_types: List[str] = []
    for value in values:
        if not value:
            continue
        for item in value.split(","):
            item = item.strip()
            if item:
                model_types.append(item)
    if not model_types:
        return None
    return sorted(set(model_types))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download model assets.")
    parser.add_argument(
        "--model-type",
        dest="model_types",
        action="append",
        help=(
            "Model type(s) to pre-download, e.g. flux2 or sdxl. "
            "Repeat the flag or use commas for multiple values. "
            "If omitted, downloads all models."
        ),
    )
    parser.add_argument(
        "--list-model-types",
        dest="list_model_types",
        action="store_true",
        help="List supported model types and exit.",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        default=settings.hf_token,
        help="Hugging Face token for gated models (defaults to HF_TOKEN env).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if args.list_model_types:
        available = sorted({model_type.value for model_type in ModelType})
        logger.info("Supported model types:")
        for model_type in available:
            logger.info("  %s", model_type)
        return 0

    selected_model_types = _parse_model_types(args.model_types)
    if selected_model_types:
        available = {model_type.value for model_type in ModelType}
        unknown = [model for model in selected_model_types if model not in available]
        if unknown:
            available_sorted = sorted(available)
            logger.error("Unknown model type(s): %s", ", ".join(unknown))
            logger.info("Available model types:")
            for model_type in available_sorted:
                logger.info("  %s", model_type)
            return 1
        pipeline_configs = []
        for model_type in selected_model_types:
            try:
                config = get_pipeline_config(model_type)
            except Exception as e:
                logger.error("Failed to load model type %s: %s", model_type, e)
                logger.info("Install optional dependencies for %s and try again.", model_type)
                return 1
            if config:
                pipeline_configs.append(config)
    else:
        pipeline_configs = _collect_pipeline_configs()

    tasks = _build_download_tasks(pipeline_configs)
    logger.info("Download tasks: %d", len(tasks))

    from ..services.pipeline_manager import PipelineManager

    manager = PipelineManager()
    total_time = 0.0
    for task in tasks:
        total_time += manager._download_model_if_needed(
            task.repo_id,
            args.hf_token,
            allow_patterns=task.allow_patterns,
            ignore_patterns=task.ignore_patterns,
        )

    logger.info("Total download time: %.1fs", total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
