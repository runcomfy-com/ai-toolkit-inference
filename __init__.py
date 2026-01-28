"""
ai-toolkit-inference - ComfyUI custom nodes integration.

When placed in ComfyUI/custom_nodes/, this package registers inference nodes
that wrap ai-toolkit-inference pipelines.
"""

import os
import sys

# Add repo root to sys.path so src.* imports work
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import and re-export ComfyUI node mappings from comfyui_nodes subpackage
try:
    # When loaded as a proper package (common in ComfyUI), relative import works.
    from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    # When this repo root is on sys.path and imported as a top-level module (e.g. pytest),
    # the relative import has no parent package; fall back to an absolute import.
    from comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
