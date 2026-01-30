"""
CLI for pre-downloading model assets.

Examples:
  python -m src.cli.download_models --base-model "black-forest-labs/FLUX.2-dev"
  python -m src.cli.download_models
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..config import settings
from ..pipelines import PIPELINE_REGISTRY
from ..services.download_config import get_download_config
from ..services.pipeline_manager import PipelineManager


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
    configs = [pipeline_cls.CONFIG for pipeline_cls in PIPELINE_REGISTRY.values()]
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


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download model assets.")
    parser.add_argument(
        "--base-model",
        dest="base_model",
        help="Hugging Face repo ID to pre-download. If omitted, downloads all models.",
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

    pipeline_configs = _collect_pipeline_configs()

    if args.base_model:
        pipeline_configs = [cfg for cfg in pipeline_configs if cfg.base_model == args.base_model]
        if not pipeline_configs:
            available = sorted({cfg.base_model for cfg in _collect_pipeline_configs()})
            logger.error("Unknown base model: %s", args.base_model)
            logger.info("Available base models:")
            for model in available:
                logger.info("  %s", model)
            return 1

    tasks = _build_download_tasks(pipeline_configs)
    logger.info("Download tasks: %d", len(tasks))

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
