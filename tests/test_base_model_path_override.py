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
    incomplete.write_bytes(b"x" * 1024)

    monkeypatch.setattr(flux2, "AITK_HF_STALL_MB_PER_S", 0.5)
    monkeypatch.setattr(flux2, "AITK_HF_STALL_TICKS", 1)
    monkeypatch.setattr(flux2, "AITK_HF_STALL_ABORT_SECONDS", 0.02)

    stop_event = threading.Event()
    abort_event = threading.Event()
    logs = []
    watcher = threading.Thread(
        target=flux2._watch_hf_download,
        args=(stop_event, abort_event, str(blobs), "owner/repo", 1024, 0.01, logs.append),
    )
    watcher.start()
    deadline = time.time() + 1
    while time.time() < deadline and not abort_event.is_set():
        time.sleep(0.01)
    stop_event.set()
    watcher.join(timeout=1)

    assert abort_event.is_set()
    assert any("hf_download_ABORTING" in line for line in logs)
