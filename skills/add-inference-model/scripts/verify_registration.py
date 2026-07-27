#!/usr/bin/env python3
"""Phase 3 CPU gate: does a newly-added model wire up correctly, with and
without an ai-toolkit checkout, before spending a GPU?

Run from the ai-toolkit-inference repo root, twice:

    python skills/add-inference-model/scripts/verify_registration.py krea2 krea2_turbo ...
    AI_TOOLKIT_PATH=/nonexistent python skills/.../verify_registration.py krea2 ...

The second run proves the FastAPI startup path survives a missing checkout
(src/api/v1/inference.py imports PIPELINE_REGISTRY eagerly, no try/except).

Exit code is non-zero if any check fails. This does NOT download weights or
need a GPU; it checks registration, config sanity, and ComfyUI wiring.
"""

import importlib
import sys


def main(model_ids: list[str]) -> int:
    if not model_ids:
        print("usage: verify_registration.py <model_id> [<model_id> ...]")
        return 2

    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = ""):
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}{'  -> ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print("== registry / startup ==")
    from src.schemas.models import get_supported_models
    from src.pipelines import get_pipeline_class, get_pipeline_config

    supported = get_supported_models()
    for mid in model_ids:
        check(f"{mid} in get_supported_models()", mid in supported)
        cls = get_pipeline_class(mid)
        check(f"{mid} resolves to a pipeline class", cls is not None)
        if cls is not None:
            check(f"{mid} CONFIG.model_type matches", cls.CONFIG.model_type.value == mid)

    # the eager path that breaks the whole catalogue if a module-scope import fails
    try:
        from src.pipelines import PIPELINE_REGISTRY  # noqa: F401
        importlib.import_module("src.api.v1.inference")
        check("PIPELINE_REGISTRY builds + src.api.v1.inference imports", True)
    except Exception as e:  # pragma: no cover
        check("PIPELINE_REGISTRY builds + src.api.v1.inference imports", False, repr(e))

    print("== config sanity ==")
    for mid in model_ids:
        cfg = get_pipeline_config(mid)
        if cfg is None:
            check(f"{mid} has a config", False)
            continue
        check(f"{mid} resolution_divisor > 0", cfg.resolution_divisor > 0,
              str(cfg.resolution_divisor))
        check(f"{mid} default_steps >= 1", cfg.default_steps >= 1, str(cfg.default_steps))
        check(f"{mid} guidance_scale >= 0", cfg.default_guidance_scale >= 0,
              str(cfg.default_guidance_scale))

    print("== download config ==")
    from src.schemas.models import ModelType
    from src.services.download_config import get_download_config

    by_repo: dict[str, set] = {}
    for mid in model_ids:
        dc = get_download_config(ModelType(mid))
        for extra in dc.extras:
            by_repo.setdefault(extra.repo_id, set()).add(
                tuple(extra.allow_patterns) if extra.allow_patterns else None
            )
    for repo, patterns in by_repo.items():
        check(f"shared repo {repo} has consistent allow_patterns", len(patterns) == 1,
              str(patterns))

    print("== ComfyUI wiring ==")
    try:
        import inspect
        from comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        from comfyui_nodes.rc_latent_workflow import RCAITKLoadPipeline

        all_pipelines = RCAITKLoadPipeline.ALL_PIPELINES
        ctor_src = inspect.getsource(RCAITKLoadPipeline)
        for mid in model_ids:
            check(f"{mid} in ALL_PIPELINES", mid in all_pipelines)
            check(f"{mid} in ctor_map", f'"{mid}": lambda' in ctor_src)
        # node-level: any node whose MODEL_ID is one of ours has matching RESOLUTION_STEP
        for node_name, node in NODE_CLASS_MAPPINGS.items():
            mid = getattr(node, "MODEL_ID", None)
            if mid in model_ids:
                check(f"{node_name} registered display name",
                      node_name in NODE_DISPLAY_NAME_MAPPINGS)
                cfg = get_pipeline_config(mid)
                if cfg is not None:
                    check(f"{node_name} RESOLUTION_STEP == resolution_divisor",
                          getattr(node, "RESOLUTION_STEP", None) == cfg.resolution_divisor,
                          f"{getattr(node, 'RESOLUTION_STEP', None)} vs {cfg.resolution_divisor}")
    except Exception as e:  # pragma: no cover
        check("ComfyUI nodes import", False, repr(e))

    print()
    if fails:
        print(f"FAILED: {len(fails)} check(s): {fails}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
