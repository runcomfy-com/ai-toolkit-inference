"""
FLUX.2 pipeline implementation with LoRA hotswap support.

Note: FLUX.2 requires ai-toolkit for custom pipeline and model classes.
This version adds minimal changes for LoRA switching while keeping
the exact same inference behavior as the original.
"""

import gc
import glob
import json
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import torch
from PIL import Image

from ..config import settings
from ..schemas.models import ModelType
from .base import BasePipeline, LoraMergeMethod, PipelineConfig

logger = logging.getLogger(__name__)


def _safe_file_info(path: Optional[str]) -> str:
    """Return non-secret file diagnostics for logs."""
    if not path:
        return "path=<none>"
    try:
        stat = os.stat(path)
        return f"path={path} exists=True size_bytes={stat.st_size}"
    except FileNotFoundError:
        return f"path={path} exists=False"
    except OSError as exc:
        return f"path={path} stat_error={type(exc).__name__}: {exc}"


@contextmanager
def _timed_step(name: str):
    start = time.perf_counter()
    logger.info(f"[FLUX2_LOAD] {name} START")
    try:
        yield
    except Exception:
        logger.exception(f"[FLUX2_LOAD] {name} FAILED after {time.perf_counter() - start:.3f}s")
        raise
    else:
        logger.info(f"[FLUX2_LOAD] {name} END elapsed={time.perf_counter() - start:.3f}s")

# Environment variable to enable aggressive memory cleanup (default: off for faster iterative runs)
AITK_AGGRESSIVE_CLEANUP = os.environ.get("AITK_AGGRESSIVE_CLEANUP", "").lower() in ("1", "true", "yes")
# Opt-in: keep a full CPU snapshot of transformer weights for fast LoRA switching.
# Default off to avoid duplicating RAM.
AITK_FLUX2_HOTSWAP = os.environ.get("AITK_FLUX2_HOTSWAP", "").lower() in ("1", "true", "yes")

# --- HF download observability (issue #23) -----------------------------------
# hf_hub_download exposes no progress callback and its tqdm bar is swallowed in the
# RunComfy/ComfyUI server context, so a frozen multi-GB download looks identical to a
# slow one in our logs. We watch the on-disk *.incomplete blob from a side thread and
# log throughput + a STALLED line so a hang is distinguishable from progress.
AITK_HF_PROGRESS = os.environ.get("AITK_HF_PROGRESS", "1").lower() not in ("0", "false", "no")
# Seconds between progress samples.
AITK_HF_PROGRESS_INTERVAL = float(os.environ.get("AITK_HF_PROGRESS_INTERVAL", "5"))
# Below this throughput a sample counts as "no progress"; N consecutive samples -> STALLED.
AITK_HF_STALL_MB_PER_S = float(os.environ.get("AITK_HF_STALL_MB_PER_S", "0.5"))
AITK_HF_STALL_TICKS = int(os.environ.get("AITK_HF_STALL_TICKS", "3"))
# Hard-abort an HF download subprocess after this many seconds without progress.
# Set to 0 to disable aborts while keeping progress logs.
AITK_HF_STALL_ABORT_SECONDS = float(os.environ.get("AITK_HF_STALL_ABORT_SECONDS", "120"))
# Abort once the blob is fully downloaded but hf_hub_download still hasn't returned (stuck
# verifying/moving into cache — the issue #23 100% hang). On abort the byte-complete blob is
# salvaged automatically, so this can be fairly short; it only needs to exceed a healthy
# finalize. 0 disables (and disables auto-salvage of finalize hangs).
AITK_HF_FINALIZE_ABORT_SECONDS = float(os.environ.get("AITK_HF_FINALIZE_ABORT_SECONDS", "90"))
# Download strategy. Default is Xet-first (fast), with an automatic plain-HTTP fallback if
# Xet hangs (its multithreaded backend can reach 100% then never return — issue #23). Set
# AITK_HF_ROBUST_DOWNLOAD=1 to skip Xet from the start (always plain HTTP) on environments
# where Xet is known-broken and you want to avoid the one-time hang+fallback latency.
AITK_HF_ROBUST_DOWNLOAD = os.environ.get("AITK_HF_ROBUST_DOWNLOAD", "0").lower() not in ("0", "false", "no")


def _maybe_cleanup() -> None:
    """Optionally run aggressive cleanup (gc + CUDA cache clear)."""
    if not AITK_AGGRESSIVE_CLEANUP:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hf_blobs_dir(cache_dir: str, repo_id: str) -> str:
    """Path to the blobs/ dir where huggingface_hub streams *.incomplete files for a repo."""
    return os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"), "blobs")


def _watch_hf_download(
    stop_event: "threading.Event",
    abort_event: Optional["threading.Event"],
    blobs_dir: str,
    repo_id: str,
    total_bytes: Optional[int],
    interval: float,
    log: Callable[[str], None],
) -> None:
    """Poll the largest *.incomplete blob and log throughput / stall until stop_event is set.

    Best-effort side channel: watcher errors are swallowed, but a confirmed no-progress
    window can set ``abort_event`` so the parent kills the isolated download subprocess.

    Once the blob reaches its expected size the download is treated as finalizing
    (verifying/moving into cache), not stalled: no STALLED warning and never an abort.
    """
    total_str = f"{total_bytes / 1e9:.2f}GB" if total_bytes else "?GB"
    last_size, last_t, stalled = 0, time.perf_counter(), 0
    finalize_start: Optional[float] = None
    last_finalize_log = 0.0
    while not stop_event.wait(interval):
        try:
            incompletes = glob.glob(os.path.join(blobs_dir, "*.incomplete"))
            if not incompletes:
                continue
            blob_path = max(incompletes, key=os.path.getsize)
            size = os.path.getsize(blob_path)
        except OSError:
            continue
        now = time.perf_counter()
        dt = now - last_t
        speed = (size - last_size) / dt / 1e6 if dt > 0 else 0.0
        pct = f"{size / total_bytes * 100:.1f}%" if total_bytes else "?%"

        # Bytes fully on disk but hf_hub_download hasn't returned: it's finalizing
        # (verifying / moving the blob into the cache), NOT a dropped connection. 0 MB/s here
        # is expected, so this is never a "stall". But it can hang indefinitely (issue #23),
        # so abort after AITK_HF_FINALIZE_ABORT_SECONDS with an actionable error.
        if total_bytes and size >= total_bytes * 0.999:
            stalled = 0
            if finalize_start is None:
                finalize_start = now
                last_finalize_log = now
                log(
                    f"[FLUX2_LOAD] hf_download_finalizing repo_id={repo_id} "
                    f"bytes_complete={size / 1e9:.2f}GB/{total_str}; hf_hub_download is "
                    f"verifying/moving the file into the cache (not a network stall)"
                )
            finalize_elapsed = now - finalize_start
            if now - last_finalize_log >= 30:
                log(f"[FLUX2_LOAD] hf_download_finalizing repo_id={repo_id} still finalizing ~{int(finalize_elapsed)}s")
                last_finalize_log = now
            if (
                abort_event is not None
                and AITK_HF_FINALIZE_ABORT_SECONDS > 0
                and finalize_elapsed >= AITK_HF_FINALIZE_ABORT_SECONDS
            ):
                log(
                    f"[FLUX2_LOAD] hf_download_FINALIZE_TIMEOUT repo_id={repo_id} "
                    f"bytes complete but hf_hub_download did not return after ~{int(finalize_elapsed)}s "
                    f"(AITK_HF_FINALIZE_ABORT_SECONDS={AITK_HF_FINALIZE_ABORT_SECONDS:g}); aborting and "
                    f"auto-salvaging the complete blob at {blob_path} (bypasses the hung HF finalize)."
                )
                abort_event.set()
            last_size, last_t = size, now
            continue

        log(
            f"[FLUX2_LOAD] hf_download_progress repo_id={repo_id} "
            f"{size / 1e9:.2f}GB/{total_str} ({pct}) {speed:.1f}MB/s"
        )
        # Don't count the first sample as a stall (no prior baseline to compare against).
        if last_size and speed < AITK_HF_STALL_MB_PER_S:
            stalled += 1
            stalled_seconds = stalled * interval
            if stalled >= AITK_HF_STALL_TICKS:
                log(
                    f"[FLUX2_LOAD] hf_download_STALLED repo_id={repo_id} "
                    f"no_progress_for~{int(stalled_seconds)}s at {size / 1e9:.2f}GB/{total_str} "
                    f"(connection likely dropped/throttled)"
                )
            if (
                abort_event is not None
                and AITK_HF_STALL_ABORT_SECONDS > 0
                and stalled_seconds >= AITK_HF_STALL_ABORT_SECONDS
            ):
                log(
                    f"[FLUX2_LOAD] hf_download_ABORTING repo_id={repo_id} "
                    f"no_progress_for~{int(stalled_seconds)}s exceeded "
                    f"AITK_HF_STALL_ABORT_SECONDS={AITK_HF_STALL_ABORT_SECONDS:g}"
                )
                abort_event.set()
        else:
            stalled = 0
        last_size, last_t = size, now


@contextmanager
def _hf_download_progress(repo_id: str, filename: str, token: Optional[str], interval: float = AITK_HF_PROGRESS_INTERVAL):
    """Log live progress for an hf_hub_download wrapped by this context manager.

    No-op when AITK_HF_PROGRESS is disabled. Resolves the file's total size and the hub cache
    location up front (both best-effort) so the watcher thread can report a percentage.
    Yields an abort event that is set when the watched blob makes no progress for too long.
    """
    if not AITK_HF_PROGRESS:
        yield None
        return

    import huggingface_hub

    total_bytes: Optional[int] = None
    try:
        meta = huggingface_hub.get_hf_file_metadata(
            huggingface_hub.hf_hub_url(repo_id, filename), token=token
        )
        total_bytes = meta.size
    except Exception as exc:  # network/auth/etc — degrade to no-percentage progress
        logger.info(f"[FLUX2_LOAD] hf_download_progress size_lookup_failed {type(exc).__name__}: {exc}")

    try:
        cache_dir = huggingface_hub.constants.HF_HUB_CACHE
    except Exception:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    blobs_dir = _hf_blobs_dir(cache_dir, repo_id)
    logger.info(f"[FLUX2_LOAD] hf_cache blobs_dir={blobs_dir} (downloaded weights land here)")

    stop_event = threading.Event()
    abort_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_hf_download,
        args=(stop_event, abort_event, blobs_dir, repo_id, total_bytes, interval, logger.info),
        name=f"hf-dl-watch-{filename}",
        daemon=True,
    )
    watcher.start()
    try:
        yield abort_event
    finally:
        stop_event.set()
        watcher.join(timeout=interval + 1.0)


def _hf_subprocess_env(repo_id: str, filename: str, robust: bool) -> Dict[str, str]:
    """Build the child-process environment for an hf_hub_download subprocess.

    When ``robust`` is set, disable hf_transfer and Xet so the child uses the plain
    single-stream HTTP downloader. The child imports huggingface_hub fresh, so these env vars
    actually take effect (unlike setting them in this already-imported process).
    """
    env = os.environ.copy()
    env["AITK_HF_REPO_ID"] = repo_id
    env["AITK_HF_FILENAME"] = filename
    if robust:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        env["HF_HUB_DISABLE_XET"] = "1"
    return env


def _run_hf_hub_download_subprocess(
    repo_id: str,
    filename: str,
    token: Optional[str],
    abort_event: Optional["threading.Event"],
    robust: bool = False,
) -> str:
    """Run hf_hub_download in a killable child process.

    A stuck requests/hf_transfer read cannot be safely interrupted in-process. Isolating the
    download lets the parent abort a no-progress tail and surface a retryable error instead
    of hanging the ComfyUI execution thread indefinitely.
    """
    child_code = r'''
import json
import os
import sys
import traceback
import huggingface_hub

try:
    token = sys.stdin.read() or None
    path = huggingface_hub.hf_hub_download(
        repo_id=os.environ["AITK_HF_REPO_ID"],
        filename=os.environ["AITK_HF_FILENAME"],
        token=token,
    )
    print(json.dumps({"ok": True, "path": path}), flush=True)
except Exception as exc:
    print(json.dumps({"ok": False, "type": type(exc).__name__, "message": str(exc)}), flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
    env = _hf_subprocess_env(repo_id, filename, robust)

    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    assert proc.stdin is not None
    proc.stdin.write(token or "")
    proc.stdin.close()
    proc.stdin = None
    while proc.poll() is None:
        if abort_event is not None and abort_event.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            raise TimeoutError(
                f"hf_hub_download for {repo_id}/{filename} was aborted by the progress watcher "
                f"(no-progress or finalize timeout — see [FLUX2_LOAD] logs). Retry may resume from "
                f"cache; if it recurs, set a local base_model_path or keep AITK_HF_ROBUST_DOWNLOAD=1."
            )
        time.sleep(0.25)

    stdout, _ = proc.communicate()
    if proc.returncode != 0:
        message = stdout.strip() or f"exit_code={proc.returncode}"
        raise RuntimeError(f"hf_hub_download subprocess failed for {repo_id}/{filename}: {message}")

    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"hf_hub_download subprocess returned invalid output for {repo_id}/{filename}") from exc
    if not result.get("ok"):
        raise RuntimeError(
            f"hf_hub_download subprocess failed for {repo_id}/{filename}: "
            f"{result.get('type')}: {result.get('message')}"
        )
    return result["path"]


def _hf_cache_dir() -> str:
    try:
        import huggingface_hub
        return huggingface_hub.constants.HF_HUB_CACHE
    except Exception:
        return os.path.expanduser("~/.cache/huggingface/hub")


def _salvaged_path(repo_id: str, filename: str) -> str:
    """Deterministic aitk-managed path for a salvaged blob, on the HF cache filesystem.

    Lives under the cache root so renaming a sibling blob into it is an instant same-filesystem
    move (no multi-GB copy).
    """
    return os.path.join(_hf_cache_dir(), "aitk_salvaged", repo_id.replace("/", "--"), filename)


def _salvage_incomplete_blob(
    repo_id: str, filename: str, token: Optional[str], label: str, require_size_match: bool = False
) -> Optional[str]:
    """Recover a byte-complete *.incomplete blob when hf_hub_download's finalize hangs.

    The bytes are fully on disk; only HF's post-download finalize (verify/move/symlink) is
    stuck (issue #23, large files on slow cache filesystems). Move the complete blob to a
    stable aitk path (instant same-fs rename) and return it, so we never depend on HF's
    finalize. Returns None if no complete blob is present.

    With ``require_size_match`` the blob is only salvaged when its size can be positively
    confirmed against HF metadata — used for the speculative pre-download check so an
    in-progress/partial *.incomplete is never mistaken for a complete file.
    """
    import huggingface_hub

    blobs_dir = _hf_blobs_dir(_hf_cache_dir(), repo_id)
    try:
        incompletes = glob.glob(os.path.join(blobs_dir, "*.incomplete"))
    except OSError:
        incompletes = []
    if not incompletes:
        return None
    blob = max(incompletes, key=os.path.getsize)
    size = os.path.getsize(blob)

    expected: Optional[int] = None
    try:
        meta = huggingface_hub.get_hf_file_metadata(huggingface_hub.hf_hub_url(repo_id, filename), token=token)
        expected = meta.size
    except Exception:
        pass
    if expected is None:
        if require_size_match:
            return None  # can't confirm completeness -> don't speculatively salvage
    elif size < expected:
        logger.warning(
            f"[FLUX2_LOAD] {label} salvage skipped: blob only {size}/{expected} bytes "
            f"(download was genuinely incomplete, not a finalize hang)"
        )
        return None

    salvaged = _salvaged_path(repo_id, filename)
    os.makedirs(os.path.dirname(salvaged), exist_ok=True)
    os.replace(blob, salvaged)  # same filesystem as blobs -> atomic, instant
    logger.info(
        f"[FLUX2_LOAD] {label} salvaged byte-complete download (bypassed hung HF finalize): "
        f"{_safe_file_info(salvaged)}"
    )
    return salvaged


def _resolve_model_file(repo_or_path: str, filename: str, token: Optional[str], label: str) -> str:
    """Resolve a weight file to a local path, bypassing Hugging Face when possible.

    Resolution order:
    1. If ``repo_or_path`` is a local directory, load ``<dir>/<filename>`` directly (issue #23
       workaround).
    2. If we previously salvaged this file (a prior download hang), reuse it — no download.
    3. If a byte-complete blob is already on disk, use it directly (skips re-invoking Xet).
    4. Otherwise download: Xet-first (fast), and on a hang salvage the bytes or fall back to
       a plain-HTTP retry. Fully automatic — no manual file copying.
    """
    if os.path.isdir(repo_or_path):
        local_file = os.path.join(repo_or_path, filename)
        if not os.path.isfile(local_file):
            raise FileNotFoundError(
                f"{label}: '{filename}' not found in local model dir '{repo_or_path}' "
                f"(expected {local_file})"
            )
        logger.info(f"[FLUX2_LOAD] {label} using local file, skipping HF download: {_safe_file_info(local_file)}")
        return local_file

    # Reuse a previously salvaged copy (a prior download hang) — skips the download entirely.
    salvaged = _salvaged_path(repo_or_path, filename)
    if os.path.isfile(salvaged):
        logger.info(f"[FLUX2_LOAD] {label} using previously salvaged file: {_safe_file_info(salvaged)}")
        return salvaged

    # If a byte-complete blob is already on disk (e.g. a prior Xet download that fetched every
    # chunk but hung in download_files() before the cache move), use it directly. xet_get() has
    # no "already complete -> return" early-exit like http_get(), so re-invoking the download
    # would just re-process the complete file and hang again. This skips that entirely.
    pre = _salvage_incomplete_blob(repo_or_path, filename, token, label, require_size_match=True)
    if pre is not None:
        return pre

    # Attempt 1: default strategy (Xet if the repo supports it — fastest), unless the operator
    # forced robust mode. Each attempt gets its own progress watcher / abort event.
    try:
        return _download_once(repo_or_path, filename, token, label, robust=AITK_HF_ROBUST_DOWNLOAD)
    except TimeoutError:
        # The download hung (commonly Xet reaching 100% then never returning). First try the
        # bytes it already wrote; if incomplete, fall back to a plain-HTTP retry that resumes
        # and finalizes reliably.
        recovered = _salvage_incomplete_blob(repo_or_path, filename, token, label)
        if recovered is not None:
            return recovered
        if AITK_HF_ROBUST_DOWNLOAD:
            raise  # attempt 1 was already plain HTTP; nothing faster-but-reliable to fall back to
        logger.warning(
            f"[FLUX2_LOAD] {label} download hung and bytes were incomplete; "
            f"retrying with HF_HUB_DISABLE_XET=1 (plain HTTP)"
        )
        return _download_once(repo_or_path, filename, token, label, robust=True)


def _download_once(repo_id: str, filename: str, token: Optional[str], label: str, robust: bool) -> str:
    """One download attempt with the progress/stall watcher attached.

    ``robust`` forces the plain-HTTP downloader (Xet/hf_transfer disabled) in the subprocess.
    """
    logger.info(
        f"[FLUX2_LOAD] {label} hf_hub_download repo_id={repo_id} filename={filename} "
        f"token_present={bool(token)} robust={robust}"
    )
    import huggingface_hub

    with _hf_download_progress(repo_id, filename, token) as abort_event:
        if abort_event is None:
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return _run_hf_hub_download_subprocess(repo_id, filename, token, abort_event, robust=robust)


# Scheduler config from ai-toolkit
FLUX2_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


class Flux2Pipeline(BasePipeline):
    """
    FLUX.2 pipeline with LoRA hotswap support.

    Uses ai-toolkit's custom Flux2Pipeline and Flux2 transformer.

    Key alignment points:
    - Custom model loading from ai-toolkit
    - LoRA merged before moving to GPU
    - Mistral text encoder with quantization
    - Resolution divisor: 32
    - Original weights saved for fast LoRA switching
    """

    CONFIG = PipelineConfig(
        model_type=ModelType.FLUX2,
        base_model="black-forest-labs/FLUX.2-dev",
        resolution_divisor=16,
        default_steps=25,
        default_guidance_scale=4.0,
        lora_merge_method=LoraMergeMethod.CUSTOM,  # Manual LoRA merge with hotswap
    )

    # Model/pipeline variants (overridden by Klein subclasses)
    TRANSFORMER_FILENAME = "flux2-dev.safetensors"
    TEXT_ENCODER_REPO = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    TEXT_ENCODER_TYPE = "mistral"  # "mistral" or "qwen"
    VAE_REPO = None  # Optional separate VAE repo (defaults to base model)
    IS_GUIDANCE_DISTILLED = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self._lora_path = None
        self._lora_scale = 1.0
        # Store original transformer weights (on CPU) for fast LoRA switching
        self._original_transformer_state: Optional[Dict[str, torch.Tensor]] = None
        # Timing details for pipeline loading
        self.timings: Dict[str, float] = {}

    def _get_flux2_params(self):
        from extensions_built_in.diffusion_models.flux2.src.model import Flux2Params
        return Flux2Params()

    def load(self, lora_paths: list, lora_scale: float = 1.0):
        """Load the FLUX.2 pipeline with LoRA merged."""
        logger.info(f"Flux2Pipeline.load() called with lora_paths={lora_paths}, lora_scale={lora_scale}")

        # Extract first LoRA path (string or dict MoE format)
        if lora_paths:
            first_lora = lora_paths[0]
            self._lora_path = list(first_lora.values())[0] if isinstance(first_lora, dict) else first_lora

        self._lora_scale = lora_scale

        logger.info(f"Loading FLUX.2 with lora_path={self._lora_path}, lora_scale={self._lora_scale}")
        self._load_pipeline()

    def _load_pipeline(self):
        """Load FLUX.2 pipeline using ai-toolkit components."""
        total_start = time.perf_counter()
        logger.info(
            f">>> _load_pipeline() START: self._lora_path={self._lora_path}, "
            f"self._lora_scale={self._lora_scale}, device={self.device}, dtype={self.dtype}, "
            f"base_model={self.CONFIG.base_model}, base_model_path_override={self.base_model_path!r}, "
            f"transformer_file={self.TRANSFORMER_FILENAME}, "
            f"text_encoder={self.TEXT_ENCODER_REPO}, text_encoder_type={self.TEXT_ENCODER_TYPE}, "
            f"vae_repo={self.VAE_REPO or self.CONFIG.base_model}, hf_token_present={bool(self.hf_token)}"
        )

        # Reset timings
        self.timings = {}

        # Add ai-toolkit to path (configurable via AI_TOOLKIT_PATH env var)
        ai_toolkit_path = settings.ai_toolkit_path
        logger.info(
            f"[FLUX2_LOAD] ai_toolkit_path={ai_toolkit_path} "
            f"exists={os.path.exists(ai_toolkit_path)} in_sys_path={ai_toolkit_path in sys.path}"
        )
        if os.path.exists(ai_toolkit_path) and ai_toolkit_path not in sys.path:
            sys.path.insert(0, ai_toolkit_path)
            logger.info("[FLUX2_LOAD] inserted ai_toolkit_path into sys.path")

        try:
            with _timed_step("import_dependencies"):
                # huggingface_hub is imported lazily by _resolve_model_file / _hf_download_progress.
                from safetensors.torch import load_file

                # Try to import ai-toolkit components
                try:
                    from extensions_built_in.diffusion_models.flux2.src.autoencoder import (
                        AutoEncoder,
                        AutoEncoderParams,
                    )
                    from extensions_built_in.diffusion_models.flux2.src.model import Flux2
                    from extensions_built_in.diffusion_models.flux2.src.pipeline import (
                        Flux2Pipeline as AITKFlux2Pipeline,
                    )
                    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
                except ImportError:
                    raise ImportError(
                        f"FLUX.2 requires ai-toolkit. Please ensure AI_TOOLKIT_PATH ({ai_toolkit_path}) is valid."
                    )

            # Clear GPU memory before loading
            with _timed_step("pre_load_cleanup"):
                _maybe_cleanup()

            # Honor an optional local model dir override (issue #23). When base_model_path is a
            # local directory, transformer/VAE weights load from disk instead of downloading.
            base_model_path = self._resolve_base_model_source()

            # 1. Load Transformer
            t_start = time.perf_counter()
            logger.info("Loading FLUX.2 Transformer")
            with _timed_step("transformer_init_meta"):
                with torch.device("meta"):
                    transformer = Flux2(self._get_flux2_params())

            # Download transformer weights (or load from the local override dir)
            transformer_filename = self.TRANSFORMER_FILENAME
            with _timed_step("transformer_hf_hub_download"):
                transformer_path = _resolve_model_file(
                    base_model_path, transformer_filename, self.hf_token, "transformer"
                )
                logger.info(f"[FLUX2_LOAD] transformer_file {_safe_file_info(transformer_path)}")

            with _timed_step("transformer_safetensors_load_cpu"):
                state_dict = load_file(transformer_path, device="cpu")
                logger.info(f"[FLUX2_LOAD] transformer_state_dict_keys={len(state_dict)}")
            with _timed_step("transformer_cast_dtype"):
                for key in state_dict:
                    state_dict[key] = state_dict[key].to(self.dtype)
            with _timed_step("transformer_load_state_dict"):
                transformer.load_state_dict(state_dict, assign=True)
            self.timings["load_transformer"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] load_transformer: {self.timings['load_transformer']:.3f}s")

            # Optional: Save original weights before LoRA merge (RAM-heavy).
            if AITK_FLUX2_HOTSWAP:
                t_start = time.perf_counter()
                logger.info("Saving original transformer weights for LoRA switching (AITK_FLUX2_HOTSWAP=1)")
                self._original_transformer_state = {k: v.clone() for k, v in transformer.state_dict().items()}
                self.timings["save_original_weights"] = time.perf_counter() - t_start
                logger.info(f"Saved {len(self._original_transformer_state)} original weight tensors")
                logger.info(f"[TIMING] save_original_weights: {self.timings['save_original_weights']:.3f}s")
            else:
                self._original_transformer_state = None
                logger.info(
                    "Skipping original transformer weight snapshot (AITK_FLUX2_HOTSWAP not enabled); "
                    "LoRA switching will require full reload."
                )

            # 2. Load and merge LoRA before moving to GPU
            t_start = time.perf_counter()
            logger.info(
                f"LoRA check: lora_path={self._lora_path}, exists={os.path.exists(self._lora_path) if self._lora_path else 'N/A'}"
            )
            if self._lora_path and os.path.exists(self._lora_path):
                logger.info(f"Loading and merging LoRA: {_safe_file_info(self._lora_path)} with scale={self._lora_scale}")
                with _timed_step("lora_merge_total"):
                    self._merge_adapter_to_transformer(transformer, self._lora_path, self._lora_scale)
            elif self._lora_path:
                logger.error(f"LoRA file not found: {self._lora_path}")
            else:
                logger.warning("No LoRA path provided, using base model only")
            self.timings["merge_lora"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] merge_lora: {self.timings['merge_lora']:.3f}s")

            t_start = time.perf_counter()
            with _timed_step("transformer_to_device"):
                transformer.to(self.device, dtype=self.dtype)
                transformer.eval()
            self.timings["transformer_to_gpu"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] transformer_to_gpu: {self.timings['transformer_to_gpu']:.3f}s")

            # Clear memory after transformer load
            with _timed_step("post_transformer_cleanup"):
                _maybe_cleanup()

            # 3. Load VAE
            t_start = time.perf_counter()
            logger.info("Loading FLUX.2 VAE")
            with _timed_step("vae_init_meta"):
                with torch.device("meta"):
                    vae = AutoEncoder(AutoEncoderParams())

            vae_filename = "ae.safetensors"
            vae_repo = self.VAE_REPO or base_model_path
            with _timed_step("vae_hf_hub_download"):
                vae_path = _resolve_model_file(vae_repo, vae_filename, self.hf_token, "vae")
                logger.info(f"[FLUX2_LOAD] vae_file {_safe_file_info(vae_path)}")

            with _timed_step("vae_safetensors_load_cpu"):
                vae_state_dict = load_file(vae_path, device="cpu")
                logger.info(f"[FLUX2_LOAD] vae_state_dict_keys={len(vae_state_dict)}")
            with _timed_step("vae_cast_dtype"):
                for key in vae_state_dict:
                    vae_state_dict[key] = vae_state_dict[key].to(self.dtype)
            with _timed_step("vae_load_state_dict"):
                vae.load_state_dict(vae_state_dict, assign=True)
            with _timed_step("vae_to_device"):
                vae.to(self.device, dtype=self.dtype)
                vae.eval()
            self.timings["load_vae"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] load_vae: {self.timings['load_vae']:.3f}s")

            # Clear memory after VAE load
            with _timed_step("post_vae_cleanup"):
                _maybe_cleanup()

            # 4. Load Text Encoder
            t_start = time.perf_counter()
            if self.TEXT_ENCODER_TYPE == "mistral":
                from transformers import AutoProcessor, Mistral3ForConditionalGeneration

                logger.info(f"Loading Mistral Text Encoder: {self.TEXT_ENCODER_REPO}")
                with _timed_step("mistral_text_encoder_from_pretrained"):
                    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                        self.TEXT_ENCODER_REPO,
                        torch_dtype=self.dtype,
                        token=self.hf_token,
                    )
            elif self.TEXT_ENCODER_TYPE == "qwen":
                from transformers import Qwen3ForCausalLM

                logger.info(f"Loading Qwen3 Text Encoder: {self.TEXT_ENCODER_REPO}")
                with _timed_step("qwen_text_encoder_from_pretrained"):
                    text_encoder = Qwen3ForCausalLM.from_pretrained(
                        self.TEXT_ENCODER_REPO,
                        torch_dtype=self.dtype,
                        token=self.hf_token,
                    )
            else:
                raise ValueError(f"Unsupported text encoder type: {self.TEXT_ENCODER_TYPE}")
            self.timings["load_text_encoder"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] load_text_encoder: {self.timings['load_text_encoder']:.3f}s")

            # 5. Apply qfloat8 quantization (optional, currently disabled)
            t_start = time.perf_counter()

            # logger.info("Applying qfloat8 quantization to text encoder")
            # qtype = get_qtype("qfloat8")
            # quantize(text_encoder, weights=qtype)
            # freeze(text_encoder)

            with _timed_step("text_encoder_to_device"):
                text_encoder.to(self.device)
                text_encoder.eval()
            self.timings["text_encoder"] = time.perf_counter() - t_start
            logger.info(f"[TIMING] text_encoder: {self.timings['text_encoder']:.3f}s")

            # 6. Load Tokenizer
            if self.TEXT_ENCODER_TYPE == "mistral":
                from transformers import AutoProcessor

                with _timed_step("mistral_tokenizer_from_pretrained"):
                    tokenizer = AutoProcessor.from_pretrained(
                        self.TEXT_ENCODER_REPO,
                        token=self.hf_token,
                    )
            elif self.TEXT_ENCODER_TYPE == "qwen":
                from transformers import Qwen2Tokenizer

                with _timed_step("qwen_tokenizer_from_pretrained"):
                    tokenizer = Qwen2Tokenizer.from_pretrained(
                        self.TEXT_ENCODER_REPO,
                        token=self.hf_token,
                    )
            else:
                raise ValueError(f"Unsupported text encoder type: {self.TEXT_ENCODER_TYPE}")

            # 7. Create Scheduler
            with _timed_step("scheduler_create"):
                scheduler = CustomFlowMatchEulerDiscreteScheduler(**FLUX2_SCHEDULER_CONFIG)

            # 8. Create Pipeline
            logger.info("Creating FLUX.2 Pipeline")
            with _timed_step("aitk_pipeline_create"):
                self.pipe = AITKFlux2Pipeline(
                    scheduler=scheduler,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer,
                    text_encoder_type=self.TEXT_ENCODER_TYPE,
                    is_guidance_distilled=self.IS_GUIDANCE_DISTILLED,
                )

            self.transformer = transformer
            self.vae = vae
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer

            with _timed_step("post_pipeline_cleanup"):
                _maybe_cleanup()

            logger.info(f"FLUX.2 pipeline loaded successfully total_elapsed={time.perf_counter() - total_start:.3f}s timings={self.timings}")

        except Exception as e:
            logger.error(f"Failed to load FLUX.2 pipeline after {time.perf_counter() - total_start:.3f}s: {e}")
            raise

    def _merge_adapter_to_transformer(self, transformer, lora_path: str, lora_scale: float = 1.0):
        """Merge adapter weights (LoRA or LoKR) into transformer on GPU.

        Auto-detects the adapter format based on key names in the safetensors file:
        - Standard LoRA (lora_A/lora_B keys): merged via W += scale * (B @ A)
        - LoKR (lokr_w1/lokr_w2 keys): merged via W += scale * kron(w1, w2)
        """
        from safetensors.torch import load_file

        # Ensure merge happens on GPU when available
        try:
            param_device = next(transformer.parameters()).device
        except StopIteration:
            param_device = torch.device("cpu")
        if param_device.type != "cuda":
            logger.info("Moving transformer to GPU for adapter merge")
            transformer.to(self.device, dtype=self.dtype)
            transformer.eval()
            try:
                param_device = next(transformer.parameters()).device
            except StopIteration:
                param_device = torch.device("cpu")
        logger.info(f"Adapter merge device: {param_device}")
        # TODO: If OOMs appear in smaller GPUs, consider chunked merge or reintroduce periodic cache clears behind a flag.

        logger.info(f"Adapter file info: {_safe_file_info(lora_path)}")
        with _timed_step("adapter_safetensors_load_cpu"):
            lora_state_dict = load_file(lora_path, device="cpu")
        logger.info(f"Loaded adapter file with {len(lora_state_dict)} keys")

        lora_keys_sample = list(lora_state_dict.keys())[:5]
        logger.info(f"Sample adapter keys (before conversion): {lora_keys_sample}")

        # Remove diffusion_model. prefix
        with _timed_step("adapter_key_convert_and_cast"):
            converted_sd = {}
            for key, value in lora_state_dict.items():
                new_key = key.replace("diffusion_model.", "")
                converted_sd[new_key] = value.to(self.dtype)

        # Move tensors to merge device once (avoid per-layer CPU->GPU transfers)
        if param_device.type == "cuda":
            with _timed_step("adapter_tensors_to_merge_device"):
                for key, value in converted_sd.items():
                    if isinstance(value, torch.Tensor) and value.device.type != "cuda":
                        converted_sd[key] = value.to(device=param_device, dtype=self.dtype, non_blocking=True)

        converted_keys_sample = list(converted_sd.keys())[:5]
        logger.info(f"Sample adapter keys (after conversion): {converted_keys_sample}")

        # Auto-detect adapter type: LoKR uses lokr_w1/lokr_w2 keys.
        # any() short-circuits on first match; ~1μs for LoKR, ~10μs full scan for LoRA.
        has_lokr = any("lokr_w" in k for k in converted_sd)
        if has_lokr:
            logger.info("Detected LoKR adapter format")
            self._merge_lokr_to_transformer(transformer, converted_sd, lora_scale)
            return

        # --- Standard LoRA merge below ---

        # Get LoRA rank
        rank = None
        for k in converted_sd:
            if "lora_A.weight" in k:
                rank = converted_sd[k].shape[0]
                break

        if not rank:
            logger.warning("No lora_A.weight found in adapter file")
            return

        logger.info(f"LoRA rank: {rank}")

        # Log sample transformer keys
        transformer_keys_sample = list(transformer.state_dict().keys())[:5]
        logger.info(f"Sample transformer keys: {transformer_keys_sample}")

        # Manual merge (GPU-friendly): W += scale * (B @ A)
        transformer_state = transformer.state_dict()
        merged_count = 0
        merge_start = time.perf_counter()
        for key in list(converted_sd.keys()):
            if "lora_A.weight" in key:
                base_key = key.replace(".lora_A.weight", "")
                lora_a_key = key
                lora_b_key = key.replace("lora_A.weight", "lora_B.weight")

                if lora_b_key in converted_sd:
                    weight_key = base_key + ".weight"

                    if weight_key in transformer_state:
                        lora_a = converted_sd[lora_a_key]
                        lora_b = converted_sd[lora_b_key]

                        alpha_key = base_key + ".alpha"
                        alpha_val = converted_sd.get(alpha_key, None)
                        if alpha_val is None:
                            alpha = rank
                        else:
                            alpha = alpha_val.item() if isinstance(alpha_val, torch.Tensor) else float(alpha_val)

                        # Apply both internal scale (alpha/rank) and external lora_scale
                        scale = (alpha / rank) * lora_scale

                        original_weight = transformer_state[weight_key]
                        device = original_weight.device
                        dtype = original_weight.dtype
                        lora_a = lora_a.to(device=device, dtype=dtype, non_blocking=True)
                        lora_b = lora_b.to(device=device, dtype=dtype, non_blocking=True)

                        # In-place: W = W + scale * (B @ A)
                        original_weight.addmm_(lora_b, lora_a, beta=1.0, alpha=scale)
                        merged_count += 1


        with _timed_step("adapter_load_merged_state_dict"):
            transformer.load_state_dict(transformer_state, assign=True)
        logger.info(
            f"Merged {merged_count} LoRA layers with scale={lora_scale} "
            f"in {time.perf_counter() - merge_start:.3f}s"
        )

    def _merge_lokr_to_transformer(self, transformer, converted_sd, lora_scale: float = 1.0):
        """Merge LoKR (Low-Rank Kronecker product) adapter weights into transformer.

        Handles all LoKR variants produced by lycoris/ai-toolkit.
        ai-toolkit default (lokr_full_rank=true) produces variant 1:
        1. Full Kronecker: lokr_w1 + lokr_w2 (internal scale = 1.0)
        2. Decomposed w2: lokr_w1 + lokr_w2_a + lokr_w2_b (scale = alpha/rank)
        3. Decomposed both: lokr_w1_a/b + lokr_w2_a/b (scale = alpha/rank)
        4. Tucker decomposition: with lokr_t2

        Performance: The kron loop over ~112 layers takes ~20-80ms on GPU total,
        comparable to standard LoRA merge (~5-15ms). Both are negligible vs model
        loading time (~10-30s). The slight overhead comes from torch.kron allocating
        a temporary full-size tensor per layer, while LoRA's addmm_ is fully in-place.

        Args:
            transformer: The transformer model to merge weights into.
            converted_sd: Pre-loaded and key-converted state dict from safetensors.
            lora_scale: External LoRA/LoKR strength multiplier.
        """
        import time

        transformer_state = transformer.state_dict()

        # Collect unique LoKR layer base keys  (~0.1ms, negligible)
        lokr_base_keys = set()
        for key in converted_sd:
            for suffix in (".lokr_w1", ".lokr_w1_a", ".lokr_w2", ".lokr_w2_a"):
                if key.endswith(suffix):
                    lokr_base_keys.add(key[: -len(suffix)])
                    break

        logger.info(f"Found {len(lokr_base_keys)} LoKR layers to merge")

        transformer_keys_sample = list(transformer_state.keys())[:5]
        logger.info(f"Sample transformer keys: {transformer_keys_sample}")

        # Main merge loop: ~0.1-0.5ms per layer on GPU (kron + add_)
        merge_start = time.perf_counter()
        merged_count = 0
        for base_key in sorted(lokr_base_keys):
            weight_key = base_key + ".weight"
            if weight_key not in transformer_state:
                continue

            original_weight = transformer_state[weight_key]
            device = original_weight.device
            dtype = original_weight.dtype

            rank = None

            # Reconstruct w1
            w1_key = base_key + ".lokr_w1"
            w1a_key = base_key + ".lokr_w1_a"
            w1b_key = base_key + ".lokr_w1_b"

            if w1_key in converted_sd:
                w1 = converted_sd[w1_key].to(device=device, dtype=dtype)
                full_w1 = True
            elif w1a_key in converted_sd and w1b_key in converted_sd:
                w1a = converted_sd[w1a_key].to(device=device, dtype=dtype)
                w1b = converted_sd[w1b_key].to(device=device, dtype=dtype)
                w1 = w1a @ w1b
                rank = w1a.shape[1]
                full_w1 = False
            else:
                logger.debug(f"Skipping {base_key}: missing w1 components")
                continue

            # Reconstruct w2
            w2_key = base_key + ".lokr_w2"
            w2a_key = base_key + ".lokr_w2_a"
            w2b_key = base_key + ".lokr_w2_b"
            t2_key = base_key + ".lokr_t2"

            if w2_key in converted_sd:
                w2 = converted_sd[w2_key].to(device=device, dtype=dtype)
                full_w2 = True
            elif w2a_key in converted_sd and w2b_key in converted_sd:
                w2a = converted_sd[w2a_key].to(device=device, dtype=dtype)
                w2b = converted_sd[w2b_key].to(device=device, dtype=dtype)
                if t2_key in converted_sd:
                    t2 = converted_sd[t2_key].to(device=device, dtype=dtype)
                    w2 = torch.einsum("i j k l, i p, j r -> p r k l", t2, w2a, w2b)
                elif w2b.dim() > 2:
                    r_dim = w2b.shape[0]
                    w2 = w2a @ w2b.reshape(r_dim, -1)
                    w2 = w2.reshape(w2a.shape[0], *w2b.shape[1:])
                else:
                    w2 = w2a @ w2b
                if rank is None:
                    rank = w2a.shape[1]
                full_w2 = False
            else:
                logger.debug(f"Skipping {base_key}: missing w2 components")
                continue

            # Compute internal scale following lycoris convention:
            # full Kronecker (both w1/w2 full) -> scale = 1.0
            # decomposed -> scale = alpha / rank
            if full_w1 and full_w2:
                internal_scale = 1.0
            else:
                alpha_key = base_key + ".alpha"
                alpha_val = converted_sd.get(alpha_key, None)
                if alpha_val is not None:
                    alpha = alpha_val.item() if isinstance(alpha_val, torch.Tensor) else float(alpha_val)
                else:
                    alpha = float(rank) if rank else 1.0
                internal_scale = alpha / rank if rank and rank > 0 else 1.0

            # torch.kron: allocates a temporary tensor of original_weight's shape (~0.1ms per layer).
            # Slightly slower than LoRA's addmm_ (fully in-place), but still sub-millisecond.
            w1_kron = w1
            for _ in range(w2.dim() - w1_kron.dim()):
                w1_kron = w1_kron.unsqueeze(-1)

            delta_weight = torch.kron(w1_kron, w2.contiguous())
            delta_weight = delta_weight.reshape(original_weight.shape)

            # In-place: W += (internal_scale * lora_scale) * delta_weight
            original_weight.add_(delta_weight, alpha=internal_scale * lora_scale)
            merged_count += 1

        merge_elapsed = time.perf_counter() - merge_start

        # load_state_dict(assign=True): ~1-5ms, reassigns tensor references without copy
        transformer.load_state_dict(transformer_state, assign=True)
        logger.info(
            f"Merged {merged_count} LoKR layers with scale={lora_scale} "
            f"in {merge_elapsed:.3f}s"
        )

    def _restore_original_weights(self):
        """Restore transformer to original weights (before any LoRA merge)."""
        if self._original_transformer_state is None:
            logger.warning("No original weights saved")
            return

        # Get current state dict and restore from CPU cache
        state_dict = self.transformer.state_dict()
        for key in state_dict:
            if key in self._original_transformer_state:
                # copy_() handles cross-device transfer internally without extra allocation
                state_dict[key].copy_(self._original_transformer_state[key])

        _maybe_cleanup()

    def _load_lora(self, lora_paths: list, lora_scale: float = 1.0):
        """LoRA is already merged in _load_pipeline for FLUX.2."""
        pass  # Already handled

    def set_lora_scale(self, scale: float) -> bool:
        """
        Dynamically set LoRA scale by restoring original weights and re-merging.

        Args:
            scale: New LoRA scale value

        Returns:
            True if successful, False otherwise
        """
        import time

        if scale == self._lora_scale:
            return True

        if self._original_transformer_state is None:
            logger.warning("Original weights not saved, cannot change scale")
            return False

        if not self._lora_path:
            logger.warning("No LoRA loaded, cannot change scale")
            return False

        try:
            total_start = time.perf_counter()
            logger.info(f"=== set_lora_scale: {self._lora_scale} -> {scale} ===")

            # Restore original weights
            restore_start = time.perf_counter()
            self._restore_original_weights()
            restore_elapsed = time.perf_counter() - restore_start
            logger.info(f"[TIMING] restore_original_weights: {restore_elapsed:.3f}s")

            # Re-merge with new scale
            merge_start = time.perf_counter()
            self._merge_adapter_to_transformer(self.transformer, self._lora_path, scale)
            merge_elapsed = time.perf_counter() - merge_start
            logger.info(f"[TIMING] merge_lora: {merge_elapsed:.3f}s")

            self._lora_scale = scale
            total_elapsed = time.perf_counter() - total_start
            logger.info(f"[TIMING] set_lora_scale TOTAL: {total_elapsed:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to change LoRA scale: {e}")
            return False

    def switch_lora(self, lora_paths: list, lora_scale: float = 1.0) -> bool:
        """
        Switch to a new LoRA without reloading the base model.

        Args:
            lora_paths: New LoRA paths to load
            lora_scale: LoRA strength

        Returns:
            True if successful, False otherwise
        """
        import time

        if self.pipe is None or self.transformer is None:
            logger.warning("Pipeline not loaded, cannot switch LoRA")
            return False

        if self._original_transformer_state is None:
            logger.warning("Original weights not saved, cannot switch LoRA")
            return False

        if not lora_paths:
            logger.warning("No LoRA paths provided for switching")
            return False

        first_lora = lora_paths[0]
        new_lora_path = list(first_lora.values())[0] if isinstance(first_lora, dict) else first_lora

        if not os.path.exists(new_lora_path):
            logger.error(f"New LoRA file not found: {new_lora_path}")
            return False

        try:
            total_start = time.perf_counter()
            logger.info(f"=== switch_lora: {self._lora_path} -> {new_lora_path} ===")

            # Restore original weights
            restore_start = time.perf_counter()
            self._restore_original_weights()
            restore_elapsed = time.perf_counter() - restore_start
            logger.info(f"[TIMING] restore_original_weights: {restore_elapsed:.3f}s")

            # Merge new LoRA
            merge_start = time.perf_counter()
            self._merge_adapter_to_transformer(self.transformer, new_lora_path, lora_scale)
            merge_elapsed = time.perf_counter() - merge_start
            logger.info(f"[TIMING] merge_lora: {merge_elapsed:.3f}s")

            self._lora_path = new_lora_path
            self._lora_scale = lora_scale
            total_elapsed = time.perf_counter() - total_start
            logger.info(f"[TIMING] switch_lora TOTAL: {total_elapsed:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to switch LoRA: {e}")
            return False

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
        """Run FLUX.2 inference (exact same as original)."""
        negative_prompt = negative_prompt or ""
        if self.IS_GUIDANCE_DISTILLED and negative_prompt:
            logger.warning("FLUX.2 is guidance-distilled; negative_prompt will be ignored")

        # Build control_img_list from control_image/control_images
        control_img_list = []
        if control_image is not None:
            # Ensure RGB mode for ai-toolkit compatibility
            if control_image.mode != "RGB":
                control_image = control_image.convert("RGB")
            control_img_list.append(control_image)
        if control_images:
            for img in control_images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                control_img_list.append(img)

        # Keep exact same call signature as original
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            control_img_list=control_img_list if control_img_list else None,
        )

        return {"image": result.images[0]}

    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        if self.vae is not None:
            del self.vae
            self.vae = None
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self._original_transformer_state is not None:
            del self._original_transformer_state
            self._original_transformer_state = None

        self.lora_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("FLUX.2 pipeline unloaded")
