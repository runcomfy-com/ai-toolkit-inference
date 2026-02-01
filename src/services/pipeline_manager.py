"""
Pipeline manager for loading and caching model pipelines.
"""

import logging
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Dict, List, Tuple, Union
from urllib.parse import urlparse

import requests

from ..pipelines import get_pipeline_class, get_pipeline_config, PIPELINE_REGISTRY
from ..libs.url_utils import normalize_huggingface_file_url, looks_like_html_file
from .download_config import ExtraDownload, get_download_config

logger = logging.getLogger(__name__)


def _build_filtered_repo_ids() -> set[str]:
    """
    Repos that are downloaded with filters (allow/ignore) for any model type.

    If a repo can be partially downloaded, we avoid cache short-circuit for full downloads
    to ensure missing files are fetched when needed.
    """
    repo_ids: set[str] = set()
    for model_type, pipeline_cls in PIPELINE_REGISTRY.items():
        config = get_download_config(model_type)
        if config.allow_patterns or config.ignore_patterns:
            repo_ids.add(pipeline_cls.CONFIG.base_model)
        for extra in config.extras:
            if extra.allow_patterns or extra.ignore_patterns:
                repo_ids.add(extra.repo_id)
    return repo_ids


FILTERED_REPO_IDS = _build_filtered_repo_ids()


def is_url(path: str) -> bool:
    """Check if a string is a URL."""
    if not isinstance(path, str):
        return False
    return path.startswith("http://") or path.startswith("https://")


def _download_lora_from_url(url: str, cache_dir: str, timeout: int = 600) -> str:
    """
    Download LoRA file from URL to cache directory.

    Args:
        url: URL to download from
        cache_dir: Directory to cache downloaded files
        timeout: Download timeout in seconds

    Returns:
        Local file path of downloaded LoRA
    """
    original_url = url
    url, note = normalize_huggingface_file_url(url)
    if note == "huggingface_blob_to_resolve" and url != original_url:
        logger.info(f"Normalized Hugging Face URL (blob -> resolve): {original_url} -> {url}")
    elif note == "huggingface_tree_url":
        logger.warning(f"Hugging Face URL appears to be a directory view (/tree/): {url}")

    # Create hash of URL for cache filename
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]

    # Extract filename from URL or use hash
    parsed = urlparse(url)
    path_parts = parsed.path.split("/")
    filename = path_parts[-1] if path_parts[-1] else f"lora_{url_hash}.safetensors"

    # Add hash to filename to avoid conflicts
    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".safetensors"
    cached_filename = f"{name}_{url_hash}{ext}"

    # Create cache directory if not exists
    os.makedirs(cache_dir, exist_ok=True)

    local_path = os.path.join(cache_dir, cached_filename)

    class _HtmlDownloadError(RuntimeError):
        """Raised when the downloaded file is actually an HTML page."""

    def _raise_html_error() -> _HtmlDownloadError:
        suggested = None
        if isinstance(original_url, str) and "/blob/" in original_url:
            suggested = original_url.replace("/blob/", "/resolve/", 1)

        msg = (
            "Downloaded LoRA content looks like HTML, not a binary weights file. "
            "This usually happens when using a Hugging Face 'blob' URL.\n"
            f"URL: {original_url}\n"
        )
        if suggested:
            msg += f"Try this direct-download URL instead:\n{suggested}\n"
        return _HtmlDownloadError(msg)

    def _validate_lora_file(path: str) -> None:
        if not os.path.exists(path):
            raise RuntimeError("LoRA file missing after download")
        if os.path.getsize(path) == 0:
            raise RuntimeError("LoRA file is empty (0 bytes)")
        if looks_like_html_file(path):
            raise _raise_html_error()

        _, ext = os.path.splitext(path)
        if ext.lower() != ".safetensors":
            return

        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(f"Unable to import safetensors for validation: {e}")

        start = time.perf_counter()
        try:
            tensors = load_file(path)
        except Exception as e:
            raise RuntimeError(f"LoRA safetensors validation failed: {e}") from e
        del tensors
        elapsed = time.perf_counter() - start
        file_size = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"[TIMING] validate_lora: {elapsed:.3f}s ({file_size:.1f}MB)")

    # Return cached file if exists and validates
    if os.path.exists(local_path):
        logger.info(f"LoRA already cached: {local_path}")
        try:
            _validate_lora_file(local_path)
            return local_path
        except _HtmlDownloadError as e:
            logger.error(f"Cached LoRA is HTML, not retrying: {e}")
            try:
                os.remove(local_path)
            except Exception:
                pass
            raise
        except Exception as e:
            logger.warning(f"Cached LoRA failed validation, re-downloading: {e}")
            try:
                os.remove(local_path)
            except Exception:
                pass

    if note == "huggingface_tree_url":
        raise ValueError(
            "Hugging Face '/tree/' URLs are directory pages (HTML), not files. "
            "Open the file and use a '/resolve/' URL instead."
        )

    max_attempts = 3
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Downloading LoRA from {url} (attempt {attempt}/{max_attempts})...")
        start_time = time.perf_counter()
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            expected_size = None
            if response.headers.get("Content-Length"):
                try:
                    expected_size = int(response.headers.get("Content-Length"))
                except ValueError:
                    expected_size = None

            # Write to temporary file first, then rename
            temp_path = local_path + ".tmp"
            bytes_written = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bytes_written += len(chunk)

            if expected_size is not None and bytes_written != expected_size:
                raise RuntimeError(f"Incomplete download: expected {expected_size} bytes, got {bytes_written} bytes")

            os.rename(temp_path, local_path)

            _validate_lora_file(local_path)

            elapsed = time.perf_counter() - start_time
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
            logger.info(f"[TIMING] download_lora: {url} {elapsed:.3f}s ({file_size:.1f}MB)")

            return local_path
        except Exception as e:
            last_error = e
            temp_path = local_path + ".tmp"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

            if isinstance(e, _HtmlDownloadError):
                logger.error(f"LoRA download returned HTML, not retrying: {e}")
                break
            if attempt < max_attempts:
                logger.warning(f"LoRA download failed (attempt {attempt}/{max_attempts}): {url} {e}. Retrying...")
                time.sleep(2)
                continue
            break

    raise RuntimeError(f"Failed to download LoRA from {url} after {max_attempts} attempts: {last_error}")


def _download_worker(
    repo_id: str,
    token: Optional[str],
    result_queue,
    allow_patterns: Optional[list] = None,
    ignore_patterns: Optional[list] = None,
) -> None:
    """Worker function for downloading models in a separate process."""
    try:
        import os
        from huggingface_hub import snapshot_download

        # Enable progress bars in subprocess
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

        # ===== Optimization for "stuck at 99%" issue =====
        # Enable hf_transfer for faster downloads (requires: pip install hf_transfer)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        # Xet backend optimizations (new huggingface_hub uses Xet by default)
        # Sequential write mode reduces "stuck at 99%" on HDD/slow disks
        os.environ.setdefault("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "1")
        # Increase concurrent range gets for better throughput
        os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32")
        # High performance mode for better bandwidth utilization
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

        # Increase download timeout
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

        # Uncomment below to disable Xet backend if issues persist:
        # os.environ["HF_HUB_DISABLE_XET"] = "1"
        # ===== End optimization =====

        download_kwargs = {
            "repo_id": repo_id,
            "token": token,
            "max_workers": 8,  # Increased from 4 for better parallelism
        }
        if allow_patterns:
            download_kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns:
            download_kwargs["ignore_patterns"] = ignore_patterns

        snapshot_download(**download_kwargs)
        result_queue.put(("success", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


class PipelineManager:
    """
    Manages loading and caching of model pipelines.

    Supports:
    - Pipeline caching: keeps pipeline in memory between requests
    - LoRA hot-swapping: switches LoRA without reloading base model (if supported)
    - Automatic reload: reloads pipeline when base model changes or hot-swap not supported
    """

    def __init__(
        self,
        device: str = "cuda",
        offload_mode: str = "model",
        lora_download_cache_dir: str = "/tmp/lora_cache",
        # Deprecated: use offload_mode instead. Kept for backwards compatibility.
        enable_cpu_offload: Optional[bool] = None,
    ):
        self.device = device
        self.lora_download_cache_dir = lora_download_cache_dir

        # Handle backwards compatibility: enable_cpu_offload overrides offload_mode
        if enable_cpu_offload is not None:
            self.offload_mode = "model" if enable_cpu_offload else "none"
        else:
            self.offload_mode = offload_mode

        # Legacy property for backwards compatibility
        self.enable_cpu_offload = self.offload_mode in ("model", "sequential")

        # Pipeline cache
        self._cached_pipeline: Optional[Any] = None
        self._cached_model: Optional[str] = None
        self._cached_lora_paths: List[str] = []
        self._cached_lora_scale: Union[float, Dict[str, float]] = 1.0

    def get_pipeline(
        self,
        model: str,
        lora_paths: List[str],
        lora_scale: Union[float, Dict[str, float]] = 1.0,
        hf_token: Optional[str] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> Any:
        """
        Get a pipeline for the specified model and LoRA.

        Uses caching and LoRA hot-swapping when possible:
        - If same model and same LoRA: reuse cached pipeline
        - If same model but different LoRA: try hot-swap, or reload if not supported
        - If different model: unload old pipeline and load new one

        Args:
            model: Model identifier (e.g., "flux", "flex2")
            lora_paths: List of paths to the LoRA weights files
            lora_scale: LoRA strength (0.0 to 2.0) or MoE scale map
            hf_token: HuggingFace token for gated models
            timings: Optional dict to store detailed timing info

        Returns:
            The pipeline instance
        """
        if timings is None:
            timings = {}

        # Normalize lora_paths for comparison
        normalized_lora_paths = self._normalize_lora_paths(lora_paths)
        cached_normalized_paths = self._normalize_lora_paths(self._cached_lora_paths)

        # Case 1: Same model and same LoRA - reuse cached pipeline
        if (
            self._cached_pipeline is not None
            and self._cached_model == model
            and normalized_lora_paths == cached_normalized_paths
        ):
            logger.info(f"Reusing cached pipeline: model={model}")
            timings["cache_hit"] = True
            timings["cache_action"] = "reuse"

            # Update lora_scale if different
            if lora_scale != self._cached_lora_scale:
                start_time = time.perf_counter()
                if self._cached_pipeline.set_lora_scale(lora_scale):
                    self._cached_lora_scale = lora_scale
                    elapsed = time.perf_counter() - start_time
                    timings["set_lora_scale"] = elapsed
                    logger.info(f"[TIMING] set_lora_scale: {elapsed:.3f}s")
                    return self._cached_pipeline
                else:
                    # set_lora_scale not supported, need full reload with new scale
                    elapsed = time.perf_counter() - start_time
                    logger.info(f"set_lora_scale not supported for {model}, performing full reload")
                    timings["cache_action"] = "reload_scale_not_supported"
                    timings["set_lora_scale_failed"] = elapsed

                    start_time = time.perf_counter()
                    self._unload_cached_pipeline()
                    elapsed = time.perf_counter() - start_time
                    timings["unload_pipeline"] = elapsed
                    logger.info(f"[TIMING] unload_pipeline: {elapsed:.3f}s")
                    # Fall through to load new pipeline
            else:
                return self._cached_pipeline

        # Case 2: Same model but different LoRA - try hot-swap
        if self._cached_pipeline is not None and self._cached_model == model:
            logger.info(f"Same model, different LoRA: attempting hot-swap")
            timings["cache_hit"] = True

            # First resolve LoRA paths (download if URLs)
            start_time = time.perf_counter()
            resolved_lora_paths, download_time = self._resolve_lora_paths(lora_paths)
            if download_time > 0:
                timings["download_lora"] = download_time
                logger.info(f"[TIMING] download_lora: {download_time:.3f}s")

            # Try hot-swap
            start_time = time.perf_counter()
            hot_swap_success = self._cached_pipeline.switch_lora(resolved_lora_paths, lora_scale)
            elapsed = time.perf_counter() - start_time

            if hot_swap_success:
                timings["cache_action"] = "hot_swap"
                timings["switch_lora"] = elapsed
                logger.info(f"[TIMING] switch_lora (hot-swap): {elapsed:.3f}s")

                self._cached_lora_paths = lora_paths
                self._cached_lora_scale = lora_scale
                return self._cached_pipeline
            else:
                # Hot-swap not supported, need full reload
                logger.info(f"Hot-swap not supported for {model}, performing full reload")
                timings["cache_action"] = "reload_lora_not_supported"
                timings["switch_lora_failed"] = elapsed

                # Unload and reload
                start_time = time.perf_counter()
                self._unload_cached_pipeline()
                elapsed = time.perf_counter() - start_time
                timings["unload_pipeline"] = elapsed
                logger.info(f"[TIMING] unload_pipeline: {elapsed:.3f}s")

        # Case 3: Different model - unload old and load new
        elif self._cached_pipeline is not None and self._cached_model != model:
            logger.info(f"Different model: {self._cached_model} -> {model}, unloading old pipeline")
            timings["cache_hit"] = False
            timings["cache_action"] = "reload_model_changed"

            start_time = time.perf_counter()
            self._unload_cached_pipeline()
            elapsed = time.perf_counter() - start_time
            timings["unload_pipeline"] = elapsed
            logger.info(f"[TIMING] unload_pipeline: {elapsed:.3f}s")

        # Case 4: No cached pipeline
        else:
            logger.info(f"No cached pipeline, loading fresh")
            timings["cache_hit"] = False
            timings["cache_action"] = "fresh_load"

        # Load new pipeline with detailed timing
        pipeline = self._load_pipeline(model, lora_paths, lora_scale, hf_token, timings)

        # Cache the new pipeline
        self._cached_pipeline = pipeline
        self._cached_model = model
        self._cached_lora_paths = lora_paths
        self._cached_lora_scale = lora_scale

        return pipeline

    def _normalize_lora_paths(self, lora_paths: List[str]) -> List[str]:
        """Normalize LoRA paths for comparison."""
        if not lora_paths:
            return []

        result = []
        for path in lora_paths:
            if isinstance(path, dict):
                # MoE format: sort by keys for consistent comparison
                sorted_items = sorted(path.items())
                result.append(str(sorted_items))
            else:
                result.append(str(path))
        return result

    def _unload_cached_pipeline(self):
        """Unload the cached pipeline."""
        if self._cached_pipeline is not None:
            logger.info("Unloading cached pipeline")
            try:
                self._cached_pipeline.unload()
            except Exception as e:
                logger.warning(f"Error unloading cached pipeline: {e}")

            self._cached_pipeline = None
            self._cached_model = None
            self._cached_lora_paths = []
            self._cached_lora_scale = 1.0

            # Force garbage collection
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_pipeline(self, pipeline: Any = None):
        """
        Unload a pipeline to free GPU memory.

        Note: With caching enabled, this method is typically a no-op since we want
        to keep the pipeline cached. Call force_unload() to actually unload.

        Args:
            pipeline: The pipeline to unload (ignored if using cache)
        """
        # With caching, we don't unload after each request
        # The pipeline stays cached for the next request
        logger.debug("unload_pipeline called but keeping pipeline cached")

    def force_unload(self):
        """Force unload the cached pipeline to free GPU memory."""
        logger.info("Force unloading cached pipeline")
        start_time = time.perf_counter()
        self._unload_cached_pipeline()
        elapsed = time.perf_counter() - start_time
        logger.info(f"[TIMING] force_unload: {elapsed:.3f}s")
        return elapsed

    def _download_model_if_needed(
        self,
        repo_id: str,
        hf_token: Optional[str] = None,
        timeout_seconds: int = 3600,
        max_retries: int = 2,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> float:
        """
        Download model from HuggingFace Hub if not already cached.

        Args:
            repo_id: HuggingFace model repository ID
            hf_token: HuggingFace API token
            timeout_seconds: Maximum time for download before timeout
            max_retries: Number of retry attempts on failure
            allow_patterns: Only download files matching these patterns
            ignore_patterns: Skip files matching these patterns

        Returns:
            Download time in seconds (0.0 if already cached)
        """
        import os
        import multiprocessing as mp
        from huggingface_hub import snapshot_download, scan_cache_dir

        # Enable faster transfer backend if available
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        # Progress bars are enabled in the download subprocess
        allow_patterns = allow_patterns or None
        ignore_patterns = ignore_patterns or None

        # More robust cache check using scan_cache_dir
        use_cache_check = allow_patterns is None and ignore_patterns is None and repo_id not in FILTERED_REPO_IDS
        if use_cache_check:
            try:
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if repo.repo_id == repo_id:
                        logger.info(f"Model {repo_id} already in cache ({repo.size_on_disk_str})")
                        return 0.0
            except Exception as e:
                logger.debug(f"Cache check failed: {e}, proceeding with download check")

        # Download with timeout and retry
        logger.info(f"Downloading model {repo_id}...")
        if ignore_patterns:
            logger.info(f"Ignoring patterns: {ignore_patterns}")
        start_time = time.perf_counter()

        for attempt in range(max_retries + 1):
            try:
                # Use multiprocessing to enable timeout
                ctx = mp.get_context("spawn")
                result_queue = ctx.Queue()

                process = ctx.Process(
                    target=_download_worker, args=(repo_id, hf_token, result_queue, allow_patterns, ignore_patterns)
                )
                process.start()
                process.join(timeout=timeout_seconds)

                if process.is_alive():
                    # Timeout occurred
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()

                    if attempt < max_retries:
                        logger.warning(
                            f"Download timeout after {timeout_seconds}s (attempt {attempt + 1}/{max_retries + 1}), retrying..."
                        )
                        continue
                    else:
                        raise TimeoutError(f"Download timeout after {timeout_seconds}s for {repo_id}")

                # Check result
                if not result_queue.empty():
                    status, error = result_queue.get()
                    if status == "error":
                        raise RuntimeError(f"Download failed: {error}")

                # Success
                elapsed = time.perf_counter() - start_time
                logger.info(f"[TIMING] download_model ({repo_id}): {elapsed:.3f}s")
                return elapsed

            except (TimeoutError, RuntimeError) as e:
                if attempt < max_retries:
                    logger.warning(f"Download failed: {e}, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(5)  # Brief delay before retry
                    continue
                else:
                    logger.error(f"Download failed after {max_retries + 1} attempts")
                    raise

        return 0.0  # Should not reach here

    def _resolve_lora_paths(
        self,
        lora_paths: List[str],
        timings: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], float]:
        """
        Resolve LoRA paths, downloading URL-based LoRAs to cache.

        Returns:
            Tuple of (resolved_paths, total_download_time)
        """
        resolved_paths = []
        total_time = 0.0

        for path in lora_paths:
            if isinstance(path, dict):
                # MoE format: {"low": path, "high": path}
                resolved_dict = {}
                for key, value in path.items():
                    if value and is_url(value):
                        start = time.perf_counter()
                        resolved_dict[key] = _download_lora_from_url(value, self.lora_download_cache_dir)
                        total_time += time.perf_counter() - start
                    else:
                        resolved_dict[key] = value
                resolved_paths.append(resolved_dict)
            elif is_url(path):
                start = time.perf_counter()
                resolved_paths.append(_download_lora_from_url(path, self.lora_download_cache_dir))
                total_time += time.perf_counter() - start
            else:
                resolved_paths.append(path)

        return resolved_paths, total_time

    def _get_download_patterns(self, pipeline_config: Any) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Get allow/ignore patterns for model download."""
        download_config = get_download_config(pipeline_config.model_type)
        return download_config.allow_patterns, download_config.ignore_patterns

    def _get_extra_downloads(self, pipeline_config: Any) -> List[ExtraDownload]:
        """Return extra repos to download (e.g., text encoders / VAEs)."""
        return get_download_config(pipeline_config.model_type).extras

    def _download_all_parallel(
        self,
        pipeline_config: Any,
        lora_paths: List[str],
        hf_token: Optional[str],
        timings: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[str]]:
        """Download base model and LoRA URLs in parallel."""
        fp16_patterns, ignore_patterns = self._get_download_patterns(pipeline_config)
        extra_downloads = self._get_extra_downloads(pipeline_config)

        # Check if there are any URLs
        has_lora_urls = (
            any(is_url(p) if isinstance(p, str) else any(is_url(v) for v in p.values() if v) for p in lora_paths)
            if lora_paths
            else False
        )

        if not has_lora_urls:
            # Sequential download
            download_time = self._download_model_if_needed(
                pipeline_config.base_model,
                hf_token,
                allow_patterns=fp16_patterns,
                ignore_patterns=ignore_patterns,
            )
            if pipeline_config.transformer_model:
                download_time += self._download_model_if_needed(pipeline_config.transformer_model, hf_token)
            for extra in extra_downloads:
                download_time += self._download_model_if_needed(
                    extra.repo_id,
                    hf_token,
                    allow_patterns=extra.allow_patterns,
                    ignore_patterns=extra.ignore_patterns,
                )
            return download_time, lora_paths

        # Parallel download
        logger.info("Starting parallel download: base model + LoRA URLs...")
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=3 + len(extra_downloads)) as executor:
            base_future = executor.submit(
                self._download_model_if_needed,
                pipeline_config.base_model,
                hf_token,
                3600,
                2,
                fp16_patterns,
                ignore_patterns,
            )
            trans_future = (
                executor.submit(
                    self._download_model_if_needed,
                    pipeline_config.transformer_model,
                    hf_token,
                )
                if pipeline_config.transformer_model
                else None
            )
            lora_future = executor.submit(self._resolve_lora_paths, lora_paths)
            extra_futures = [
                executor.submit(
                    self._download_model_if_needed,
                    extra.repo_id,
                    hf_token,
                    3600,
                    2,
                    extra.allow_patterns,
                    extra.ignore_patterns,
                )
                for extra in extra_downloads
            ]

            base_time = base_future.result()
            trans_time = trans_future.result() if trans_future else 0.0
            resolved_paths, lora_time = lora_future.result()
            extra_times = [f.result() for f in extra_futures] if extra_futures else []

        wall_time = time.perf_counter() - start

        if timings is not None:
            timings["download_base_model"] = base_time
            if trans_future:
                timings["download_transformer"] = trans_time
            if lora_time > 0:
                timings["download_lora"] = lora_time
            if extra_times:
                timings["download_extra_models"] = sum(extra_times)
            timings["download_wall_time"] = wall_time

        logger.info(f"[TIMING] Parallel download: base={base_time:.1f}s, lora={lora_time:.1f}s, wall={wall_time:.1f}s")

        return base_time + trans_time + (sum(extra_times) if extra_times else 0.0), resolved_paths

    def predownload_pipeline_assets(self, pipeline_config: Any, hf_token: Optional[str] = None) -> float:
        """
        Pre-download base model, optional transformer, and any extra repos for a pipeline config.

        Returns:
            Total download time in seconds (0.0 if already cached).
        """
        download_config = get_download_config(pipeline_config.model_type)
        total_time = self._download_model_if_needed(
            pipeline_config.base_model,
            hf_token,
            allow_patterns=download_config.allow_patterns,
            ignore_patterns=download_config.ignore_patterns,
        )
        if pipeline_config.transformer_model:
            total_time += self._download_model_if_needed(pipeline_config.transformer_model, hf_token)
        for extra in download_config.extras:
            total_time += self._download_model_if_needed(
                extra.repo_id,
                hf_token,
                allow_patterns=extra.allow_patterns,
                ignore_patterns=extra.ignore_patterns,
            )
        return total_time

    def predownload_model(self, model: str, hf_token: Optional[str] = None) -> float:
        """Pre-download assets for a model type string (e.g., 'flux2')."""
        pipeline_config = get_pipeline_config(model)
        if not pipeline_config:
            raise ValueError(f"Unknown model: {model}")
        return self.predownload_pipeline_assets(pipeline_config, hf_token)

    def _load_pipeline(
        self,
        model: str,
        lora_paths: List[str],
        lora_scale: Union[float, Dict[str, float]],
        hf_token: Optional[str] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Load a new pipeline with detailed timing."""
        logger.info(f"Loading pipeline for model={model}, lora_paths={lora_paths}")

        # Get pipeline class (contains its own config)
        pipeline_class = get_pipeline_class(model)
        if not pipeline_class:
            raise ValueError(f"Unknown model: {model}")

        # Get pipeline config for base model info
        pipeline_config = pipeline_class.CONFIG

        # Login to HuggingFace if token provided
        if hf_token:
            start_time = time.perf_counter()
            try:
                from huggingface_hub import login

                login(token=hf_token)
                logger.info("Logged in to HuggingFace Hub")
            except Exception as e:
                logger.warning(f"Failed to login to HuggingFace Hub: {e}")
            if timings is not None:
                timings["hf_login"] = time.perf_counter() - start_time

        # Download base model and LoRA URLs in parallel
        download_time, resolved_lora_paths = self._download_all_parallel(pipeline_config, lora_paths, hf_token, timings)

        if timings is not None:
            timings["download_model"] = download_time
            if download_time > 0:
                logger.info(f"[TIMING] download_model (total): {download_time:.3f}s")

        # Create pipeline instance
        start_time = time.perf_counter()
        pipeline_instance = pipeline_class(
            device=self.device,
            offload_mode=self.offload_mode,
            hf_token=hf_token,
        )
        if timings is not None:
            timings["pipeline_init"] = time.perf_counter() - start_time
            logger.info(f"[TIMING] pipeline_init: {timings['pipeline_init']:.3f}s")

        # Load pipeline with LoRA using unified load() interface (with resolved paths)
        start_time = time.perf_counter()
        pipeline_instance.load(resolved_lora_paths, lora_scale)

        if timings is not None:
            load_time = time.perf_counter() - start_time
            timings["load_model"] = load_time
            timings["load_lora"] = 0.0  # Included in load_model
            logger.info(f"[TIMING] pipeline.load(): {load_time:.3f}s")

            # Collect pipeline-specific timing details (e.g., LoRA or component load timings)
            if pipeline_instance.timings:
                timings.update(pipeline_instance.timings)
                logger.info(f"[TIMING] pipeline_timings: {pipeline_instance.timings}")

        return pipeline_instance
