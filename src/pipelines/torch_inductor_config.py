"""Shared torch.compile inductor configuration."""

import torch

_APPLIED = False


def apply_torch_inductor_optimizations() -> None:
    """Apply shared inductor config used by Wan 2.2 pipelines."""
    global _APPLIED
    if _APPLIED:
        return
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.fx_graph_cache = True
    _APPLIED = True
