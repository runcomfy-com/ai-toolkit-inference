"""Tests for the local base_model_path override (issue #23 workaround).

These exercise the override plumbing and the file-resolution logic without loading
any real model weights:
  - get_or_load_pipeline threads base_model_path into the pipeline ctor + cache key,
    so a changed path forces a reload while an unchanged one is a cache hit.
  - BasePipeline._resolve_base_model_source() returns the override or CONFIG.base_model.
  - Flux2's _resolve_model_file() loads a local dir file directly (bypassing HF),
    raises a clear error when the file is missing, and otherwise downloads by repo id.
"""
import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


# --------------------------------------------------------------------------
# get_or_load_pipeline: base_model_path plumbing + cache invalidation
# --------------------------------------------------------------------------
@pytest.fixture
def rc(monkeypatch):
    """The rc_common module with a freshly reset pipeline cache."""
    from comfyui_nodes import rc_common

    monkeypatch.setattr(rc_common, "_PIPELINE_CACHE", {"key": None, "instance": None})
    return rc_common


def _make_fake_ctor(created):
    class FakePipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(kwargs)

        def load(self, **kw):
            self.loaded = kw

        def unload(self):
            created.append("UNLOAD")

    return FakePipe


def _load(rc, ctor, **overrides):
    kwargs = dict(
        model_id="flux",
        pipeline_ctor=ctor,
        offload_mode="none",
        hf_token=None,
        lora_paths=[],
        lora_scale=1.0,
    )
    kwargs.update(overrides)
    return rc.get_or_load_pipeline(**kwargs)


def test_base_model_path_passed_to_ctor(rc):
    created = []
    ctor = _make_fake_ctor(created)
    _load(rc, ctor, base_model_path="/models/flux-local")
    assert created[0].get("base_model_path") == "/models/flux-local"


def test_same_base_model_path_is_cache_hit(rc):
    created = []
    ctor = _make_fake_ctor(created)
    p1 = _load(rc, ctor, base_model_path="/models/flux-local")
    n_ctor = lambda: len([c for c in created if isinstance(c, dict)])
    before = n_ctor()
    p2 = _load(rc, ctor, base_model_path="/models/flux-local")
    assert p1 is p2
    assert n_ctor() == before  # no new construction


def test_changed_base_model_path_forces_reload(rc):
    created = []
    ctor = _make_fake_ctor(created)
    p1 = _load(rc, ctor, base_model_path="/models/flux-local")
    p2 = _load(rc, ctor, base_model_path="/models/OTHER")
    assert p1 is not p2
    assert "UNLOAD" in created  # previous instance was unloaded
    assert created[-1].get("base_model_path") == "/models/OTHER"


def test_empty_base_model_path_normalized_to_none(rc):
    created = []
    ctor = _make_fake_ctor(created)
    _load(rc, ctor, base_model_path="")
    assert created[0].get("base_model_path") is None


def test_base_model_path_in_cache_key():
    from comfyui_nodes.rc_common import PipelineCacheKey

    # The dataclass must carry base_model_path so it participates in cache identity.
    assert "base_model_path" in PipelineCacheKey.__dataclass_fields__


# --------------------------------------------------------------------------
# RCAITKLoadPipeline node (split loader/sampler workflow) exposes & forwards it
# --------------------------------------------------------------------------
def test_loadpipeline_node_exposes_base_model_path_input():
    from comfyui_nodes.rc_latent_workflow import RCAITKLoadPipeline

    assert "base_model_path" in RCAITKLoadPipeline.INPUT_TYPES()["optional"]


def test_loadpipeline_node_forwards_base_model_path(monkeypatch):
    from comfyui_nodes import rc_latent_workflow

    captured = {}
    monkeypatch.setattr(
        rc_latent_workflow, "get_or_load_pipeline", lambda **kw: captured.update(kw) or object()
    )

    node = rc_latent_workflow.RCAITKLoadPipeline()
    node.load(pipeline="flux2", offload_mode="none", hf_token="", base_model_path="/models/flux2-local")
    assert captured["base_model_path"] == "/models/flux2-local"


# --------------------------------------------------------------------------
# BasePipeline._resolve_base_model_source
# --------------------------------------------------------------------------
def _make_pipeline(base_model_path):
    from types import SimpleNamespace

    from src.pipelines.base import BasePipeline

    class _T(BasePipeline):
        CONFIG = SimpleNamespace(base_model="black-forest-labs/FLUX.1-dev")

        def _load_pipeline(self):  # pragma: no cover - not exercised
            ...

        def _run_inference(self, *a, **k):  # pragma: no cover - not exercised
            ...

    _T.__abstractmethods__ = frozenset()  # allow instantiation without every abstract impl
    return _T(base_model_path=base_model_path)


def test_resolve_base_model_source_defaults_to_config():
    pipe = _make_pipeline(base_model_path=None)
    assert pipe._resolve_base_model_source() == "black-forest-labs/FLUX.1-dev"


def test_resolve_base_model_source_uses_override():
    pipe = _make_pipeline(base_model_path="/models/flux-local")
    assert pipe._resolve_base_model_source() == "/models/flux-local"


# --------------------------------------------------------------------------
# Flux2._resolve_model_file: local bypass vs HF download
# --------------------------------------------------------------------------
def test_resolve_model_file_local_dir(tmp_path):
    from src.pipelines import flux2

    fname = "flux-2-klein-base-9b.safetensors"
    (tmp_path / fname).write_bytes(b"")
    got = flux2._resolve_model_file(str(tmp_path), fname, None, "transformer")
    assert got == str(tmp_path / fname)


def test_resolve_model_file_missing_raises_clear_error(tmp_path):
    from src.pipelines import flux2

    with pytest.raises(FileNotFoundError) as exc:
        flux2._resolve_model_file(str(tmp_path), "ae.safetensors", None, "vae")
    msg = str(exc.value)
    assert "ae.safetensors" in msg and "expected" in msg


def test_resolve_model_file_repo_id_downloads(monkeypatch):
    import huggingface_hub

    from src.pipelines import flux2

    # Avoid spawning the progress-watcher thread / network metadata lookup.
    monkeypatch.setattr(flux2, "AITK_HF_PROGRESS", False)
    calls = {}

    def fake_download(repo_id, filename, token):
        calls.update(repo_id=repo_id, filename=filename, token=token)
        return "/cache/blobs/abc"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    got = flux2._resolve_model_file(
        "black-forest-labs/FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors", "tok", "transformer"
    )
    assert got == "/cache/blobs/abc"
    assert calls["repo_id"] == "black-forest-labs/FLUX.2-klein-base-9B"
    assert calls["token"] == "tok"


def test_resolve_model_file_progress_path_uses_abortable_subprocess(monkeypatch):
    from src.pipelines import flux2

    abort_event = threading.Event()
    calls = {}

    @contextmanager
    def fake_progress(repo_id, filename, token):
        calls.update(progress=(repo_id, filename, token))
        yield abort_event

    def fake_subprocess(repo_id, filename, token, got_abort_event):
        calls.update(download=(repo_id, filename, token, got_abort_event))
        return "/cache/blobs/def"

    monkeypatch.setattr(flux2, "_hf_download_progress", fake_progress)
    monkeypatch.setattr(flux2, "_run_hf_hub_download_subprocess", fake_subprocess)

    got = flux2._resolve_model_file("owner/repo", "model.safetensors", "tok", "transformer")

    assert got == "/cache/blobs/def"
    assert calls["progress"] == ("owner/repo", "model.safetensors", "tok")
    assert calls["download"] == ("owner/repo", "model.safetensors", "tok", abort_event)


def test_hf_download_watcher_sets_abort_event_after_no_progress(tmp_path, monkeypatch):
    from src.pipelines import flux2

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    incomplete = blobs / "abc.incomplete"
    # Genuinely partial: 1KB present out of a 1MB file, and it never grows -> a real stall.
    incomplete.write_bytes(b"x" * 1024)

    monkeypatch.setattr(flux2, "AITK_HF_STALL_MB_PER_S", 0.5)
    monkeypatch.setattr(flux2, "AITK_HF_STALL_TICKS", 1)
    monkeypatch.setattr(flux2, "AITK_HF_STALL_ABORT_SECONDS", 0.02)

    stop_event = threading.Event()
    abort_event = threading.Event()
    logs = []
    watcher = threading.Thread(
        target=flux2._watch_hf_download,
        args=(stop_event, abort_event, str(blobs), "owner/repo", 1_000_000, 0.01, logs.append),
    )
    watcher.start()
    deadline = time.time() + 1
    while time.time() < deadline and not abort_event.is_set():
        time.sleep(0.01)
    stop_event.set()
    watcher.join(timeout=1)

    assert abort_event.is_set()
    assert any("hf_download_ABORTING" in line for line in logs)


def test_hf_download_watcher_treats_full_size_as_finalizing_not_stall(tmp_path, monkeypatch):
    """At 100% the bytes are on disk; hf_hub_download is finalizing, not stalled.

    The watcher must NOT abort (that would kill a complete download) and must NOT emit a
    STALLED warning — it logs a finalizing line instead (issue #23, 100% hang case).
    """
    from src.pipelines import flux2

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    total = 4096
    incomplete = blobs / "abc.incomplete"
    incomplete.write_bytes(b"x" * total)  # full size already on disk

    # Aggressive thresholds: if the watcher mistook this for a stall it would abort fast.
    monkeypatch.setattr(flux2, "AITK_HF_STALL_MB_PER_S", 0.5)
    monkeypatch.setattr(flux2, "AITK_HF_STALL_TICKS", 1)
    monkeypatch.setattr(flux2, "AITK_HF_STALL_ABORT_SECONDS", 0.02)

    stop_event = threading.Event()
    abort_event = threading.Event()
    logs = []
    watcher = threading.Thread(
        target=flux2._watch_hf_download,
        args=(stop_event, abort_event, str(blobs), "owner/repo", total, 0.01, logs.append),
    )
    watcher.start()
    time.sleep(0.3)  # well past the abort window if it (wrongly) treated this as a stall
    stop_event.set()
    watcher.join(timeout=1)

    assert not abort_event.is_set()
    assert not any("hf_download_STALLED" in line for line in logs)
    assert not any("hf_download_ABORTING" in line for line in logs)
    assert any("hf_download_finalizing" in line for line in logs)


def test_hf_download_watcher_finalize_timeout_aborts(tmp_path, monkeypatch):
    """A finalize that never returns must still be aborted (with its own message)."""
    from src.pipelines import flux2

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    total = 4096
    (blobs / "abc.incomplete").write_bytes(b"x" * total)

    monkeypatch.setattr(flux2, "AITK_HF_FINALIZE_ABORT_SECONDS", 0.02)

    stop_event = threading.Event()
    abort_event = threading.Event()
    logs = []
    watcher = threading.Thread(
        target=flux2._watch_hf_download,
        args=(stop_event, abort_event, str(blobs), "owner/repo", total, 0.01, logs.append),
    )
    watcher.start()
    deadline = time.time() + 1
    while time.time() < deadline and not abort_event.is_set():
        time.sleep(0.01)
    stop_event.set()
    watcher.join(timeout=1)

    assert abort_event.is_set()
    assert any("hf_download_FINALIZE_TIMEOUT" in line for line in logs)
    assert not any("hf_download_STALLED" in line for line in logs)


def test_hf_subprocess_env_forces_robust_download(monkeypatch):
    from src.pipelines import flux2

    monkeypatch.setattr(flux2, "AITK_HF_ROBUST_DOWNLOAD", True)
    env = flux2._hf_subprocess_env("owner/repo", "model.safetensors")
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["AITK_HF_REPO_ID"] == "owner/repo"
    assert env["AITK_HF_FILENAME"] == "model.safetensors"


def test_hf_subprocess_env_can_keep_fast_download(monkeypatch):
    from src.pipelines import flux2

    monkeypatch.setattr(flux2, "AITK_HF_ROBUST_DOWNLOAD", False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    env = flux2._hf_subprocess_env("owner/repo", "model.safetensors")
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in env


# --------------------------------------------------------------------------
# Auto-salvage of a byte-complete blob when HF finalize hangs (issue #23)
# --------------------------------------------------------------------------
def _seed_cache(monkeypatch, tmp_path, repo_id, etag, nbytes):
    """Point the HF cache at tmp_path and write a byte-complete *.incomplete blob."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path), raising=False)
    blobs = tmp_path / ("models--" + repo_id.replace("/", "--")) / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / f"{etag}.incomplete"
    blob.write_bytes(b"x" * nbytes)
    return blob


def test_salvage_on_finalize_timeout(tmp_path, monkeypatch):
    """A finalize-aborted download whose bytes are complete is salvaged and returned."""
    import huggingface_hub

    from src.pipelines import flux2

    repo, fname, nbytes = "owner/repo", "model.safetensors", 4096
    _seed_cache(monkeypatch, tmp_path, repo, "deadbeef", nbytes)
    monkeypatch.setattr(
        huggingface_hub, "get_hf_file_metadata",
        lambda *a, **k: SimpleNamespace(size=nbytes), raising=False,
    )

    @contextmanager
    def fake_progress(repo_id, filename, token):
        yield threading.Event()

    def boom(*a, **k):
        raise TimeoutError("finalize hang")

    monkeypatch.setattr(flux2, "_hf_download_progress", fake_progress)
    monkeypatch.setattr(flux2, "_run_hf_hub_download_subprocess", boom)

    got = flux2._resolve_model_file(repo, fname, "tok", "transformer")
    assert got == flux2._salvaged_path(repo, fname)
    assert os.path.isfile(got)
    assert os.path.getsize(got) == nbytes


def test_salvaged_file_is_reused_without_download(tmp_path, monkeypatch):
    """Once salvaged, a later load uses the salvaged file and never calls download."""
    import huggingface_hub

    from src.pipelines import flux2

    repo, fname = "owner/repo", "model.safetensors"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path), raising=False)
    salvaged = flux2._salvaged_path(repo, fname)
    os.makedirs(os.path.dirname(salvaged), exist_ok=True)
    with open(salvaged, "wb") as f:
        f.write(b"x" * 16)

    def fail_download(*a, **k):
        raise AssertionError("download must not be called when a salvaged file exists")

    monkeypatch.setattr(flux2, "_hf_download_progress", fail_download)
    got = flux2._resolve_model_file(repo, fname, "tok", "transformer")
    assert got == salvaged


def test_partial_blob_is_not_salvaged(tmp_path, monkeypatch):
    """A genuinely incomplete blob (size < expected) must not be passed off as complete."""
    import huggingface_hub

    from src.pipelines import flux2

    repo, fname = "owner/repo", "model.safetensors"
    _seed_cache(monkeypatch, tmp_path, repo, "cafef00d", nbytes=1024)
    monkeypatch.setattr(
        huggingface_hub, "get_hf_file_metadata",
        lambda *a, **k: SimpleNamespace(size=1_000_000), raising=False,  # expected >> on-disk
    )
    assert flux2._salvage_incomplete_blob(repo, fname, "tok", "transformer") is None
