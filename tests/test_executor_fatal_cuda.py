"""The worker must die on a poisoned CUDA context, not keep serving from it.

Production trace this pins (RunPod, minimax_h3, 2026-08-11): one
illegal-memory-access inside a transformer forward, after which the SAME worker
kept accepting requests and failed each one in milliseconds — some while merely
creating a text-encoder tensor — because a CUDA context that has raised an
illegal access never accepts work again. The fix is to write the failed status,
then exit so the platform replaces the worker.
"""

import pytest

from src.tasks.executor import (
    _FATAL_CUDA_MARKERS,
    _cuda_context_is_poisoned,
    _device_is_cuda,
    _die_worker,
    _is_fatal_accelerator_error,
)


class TestFatalClassifier:
    """The classifier is ADVISORY — it labels the death log line but never
    decides a kill on its own, because exception text is attacker-influenced
    (loras[].path URLs propagate remote reason text). The probe decides."""

    @pytest.mark.parametrize(
        "message",
        [
            "CUDA error: an illegal memory access was encountered",
            "CUDA error: device-side assert triggered",
            "CUDA error: unspecified launch failure",
            "CUDA error: uncorrectable ECC error encountered",
            "CUDA error: misaligned address",
        ],
    )
    def test_known_fatal_states_are_fatal(self, message):
        assert _is_fatal_accelerator_error(RuntimeError(message))

    def test_matching_is_case_insensitive(self):
        assert _is_fatal_accelerator_error(
            RuntimeError("CUDA ERROR: AN ILLEGAL MEMORY ACCESS WAS ENCOUNTERED")
        )

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("Unknown model: nope"),
            RuntimeError("CUDA out of memory. Tried to allocate 2.50 GiB"),
            RuntimeError("The size of tensor a (3) must match ..."),
            TimeoutError("Inference timed out"),
        ],
    )
    def test_ordinary_failures_are_not_fatal(self, exc):
        """Bad requests, shape bugs and OOM must NOT kill the worker — OOM in
        particular is synchronous and recoverable via empty_cache."""
        assert not _is_fatal_accelerator_error(exc)

    def test_torch_oom_type_is_excluded_even_if_message_matched(self):
        """Belt and braces: a torch OutOfMemoryError is never fatal by type,
        regardless of what its message happens to contain."""
        import torch

        oom_type = getattr(torch, "OutOfMemoryError", None)
        if oom_type is None:
            pytest.skip("this torch build has no torch.OutOfMemoryError")
        assert not _is_fatal_accelerator_error(
            oom_type("illegal memory access mentioned in passing")
        )

    def test_marker_list_matches_the_production_trace(self):
        """The incident message must always classify fatal."""
        assert any(
            m in "cuda error: an illegal memory access was encountered"
            for m in _FATAL_CUDA_MARKERS
        )


class TestDieWorker:
    def test_exits_with_code_70_after_grace(self, monkeypatch):
        calls = {}
        monkeypatch.setattr("src.tasks.executor.time.sleep", lambda s: calls.setdefault("slept", s))
        monkeypatch.setattr("src.tasks.executor.os._exit", lambda code: calls.setdefault("exit", code))

        _die_worker("test reason")

        # grace first (lets /status polls observe the failed state), then exit
        assert calls["slept"] == 3
        assert calls["exit"] == 70


class TestExecutorWiring:
    """Drive InferenceExecutor.execute through a real failure and assert the
    order of operations: mark failed -> persist -> THEN die."""

    DEVICE = "cuda:1"

    def _run(self, monkeypatch, exc, poisoned_probe=False):
        from src.tasks.executor import InferenceExecutor

        events = []
        monkeypatch.setattr(
            "src.tasks.executor._cuda_context_is_poisoned",
            lambda device: events.append(("probed", device)) or poisoned_probe,
        )
        monkeypatch.setattr(
            "src.tasks.executor._device_is_cuda", lambda device: device != "cpu"
        )
        monkeypatch.setattr("src.tasks.executor.time.sleep", lambda s: None)
        monkeypatch.setattr(
            "src.tasks.executor.os._exit", lambda code: events.append(("exit", code))
        )

        class Task:
            id = "t1"
            model = "minimax_h3"
            inputs = {"prompts": []}
            lora_paths = []
            started_at = None

            def mark_as_processing(self):
                events.append(("processing",))

            def mark_as_failed(self, error, details=None):
                events.append(("failed", error))

            def mark_as_succeeded(self, *a, **kw):
                events.append(("succeeded",))

        class Storage:
            def update(self, task):
                events.append(("stored",))

        class Manager:
            device = self.DEVICE

            def get_pipeline(self, **kw):
                raise exc

        ex = InferenceExecutor(storage=Storage(), pipeline_manager=Manager())
        # get_pipeline_config must succeed so we reach the manager and raise
        # inside the try block proper.
        ex._execute_locked(Task())
        return events

    def test_fatal_error_with_poisoned_context_dies_after_persisting(self, monkeypatch):
        events = self._run(
            monkeypatch,
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            poisoned_probe=True,
        )
        assert ("exit", 70) in events
        # the failed status was written BEFORE the exit
        failed_i = next(i for i, e in enumerate(events) if e[0] == "failed")
        exit_i = next(i for i, e in enumerate(events) if e[0] == "exit")
        assert failed_i < exit_i
        assert events[failed_i + 1] == ("stored",)

    def test_spoofed_marker_text_with_healthy_context_survives(self, monkeypatch):
        """PR #31 review: loras[].path is an arbitrary URL and remote reason
        text lands in the exception, so "500 illegal memory access" from a
        hostile server must NOT kill a worker whose context probes healthy.
        Every real fatal CUDA state is sticky and cannot pass the probe, so
        nothing real is lost by requiring the probe's confirmation."""
        events = self._run(
            monkeypatch,
            RuntimeError(
                "Failed to download LoRA: 500 illegal memory access was encountered"
            ),
            poisoned_probe=False,
        )
        assert not any(e[0] == "exit" for e in events)
        assert any(e[0] == "failed" for e in events)

    def test_ordinary_error_keeps_the_worker_alive(self, monkeypatch):
        events = self._run(monkeypatch, RuntimeError("some shape mismatch"))
        assert not any(e[0] == "exit" for e in events)
        assert any(e[0] == "failed" for e in events)

    def test_probe_catches_poisoned_context_behind_any_exception(self, monkeypatch):
        """CUDA reports errors asynchronously, so the surfaced exception can be
        arbitrary. The probe is what catches those."""
        events = self._run(
            monkeypatch, RuntimeError("some shape mismatch"), poisoned_probe=True
        )
        assert ("exit", 70) in events


class TestDeviceGating:
    """The probe must target the CONFIGURED device, and non-CUDA workers must
    never die — a DEVICE=cpu worker is healthy no matter what a stray GPU
    error message says (PR #31 review)."""

    def test_probe_receives_the_configured_device(self, monkeypatch):
        t = TestExecutorWiring()
        events = t._run(monkeypatch, RuntimeError("some shape mismatch"))
        assert ("probed", "cuda:1") in events

    def test_cpu_worker_survives_a_fatal_looking_message(self, monkeypatch):
        class CpuWiring(TestExecutorWiring):
            DEVICE = "cpu"

        events = CpuWiring()._run(
            monkeypatch,
            RuntimeError("CUDA error: an illegal memory access was encountered"),
        )
        assert not any(e[0] == "exit" for e in events)
        # ...and the probe is never even consulted
        assert not any(e[0] == "probed" for e in events)

    def test_device_is_cuda_classification(self):
        assert _device_is_cuda("cuda")
        assert _device_is_cuda("cuda:1")
        assert not _device_is_cuda("cpu")
        assert not _device_is_cuda("mps")
        assert not _device_is_cuda("not a device !!")

    def test_probe_is_a_noop_for_non_cuda_devices(self):
        assert _cuda_context_is_poisoned("cpu") is False
        assert _cuda_context_is_poisoned("not a device !!") is False
