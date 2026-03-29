"""
RCLoRAApply — custom ComfyUI node to apply a second LoRA to an already-loaded AITK_PIPELINE.

Deploy to RunComfy machine:
    /workspace/ComfyUI/custom_nodes/rc_lora_stacker/rc_lora_stacker.py
    /workspace/ComfyUI/custom_nodes/rc_lora_stacker/__init__.py

Then restart ComfyUI.
"""

import os
import hashlib
import tempfile
import requests
import torch
import safetensors.torch


def _download_lora(url_or_path: str) -> str:
    """Download a LoRA from URL to /tmp cache, or return local path as-is."""
    if not url_or_path.startswith("http"):
        return url_or_path

    # cache by URL hash to avoid re-downloading
    h = hashlib.md5(url_or_path.encode()).hexdigest()[:12]
    fname = os.path.basename(url_or_path.split("?")[0])
    cache_path = f"/tmp/lora_cache/{h}_{fname}"
    os.makedirs("/tmp/lora_cache", exist_ok=True)

    if not os.path.exists(cache_path):
        print(f"[RCLoRAApply] Downloading LoRA: {url_or_path}")
        resp = requests.get(url_or_path, timeout=120)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(resp.content)
        print(f"[RCLoRAApply] Saved to {cache_path}")
    else:
        print(f"[RCLoRAApply] Using cached LoRA: {cache_path}")

    return cache_path


def _get_transformer(pipe):
    """Extract the transformer nn.Module from an AITK_PIPELINE object."""
    # Direct attribute
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    # Nested pipeline object
    for attr in ("pipeline", "pipe", "model", "flux_pipeline"):
        obj = getattr(pipe, attr, None)
        if obj is not None and hasattr(obj, "transformer"):
            return obj.transformer
    # Dict wrapper
    if isinstance(pipe, dict):
        for key in ("pipeline", "pipe", "model"):
            obj = pipe.get(key)
            if obj is not None:
                t = _get_transformer(obj)
                if t is not None:
                    return t
    return None


def _apply_lora(transformer, lora_path: str, scale: float):
    """Merge a LoRA safetensors file into transformer weights in-place."""
    state = safetensors.torch.load_file(lora_path)

    # Strip 'diffusion_model.' prefix (RunComfy convention)
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("diffusion_model.", "")] = v

    all_keys = list(cleaned.keys())
    lora_a_keys = [k for k in all_keys if ".lora_A.weight" in k]
    alpha_keys  = [k for k in all_keys if ".alpha" in k]
    print(f"[RCLoRAApply] File: {os.path.basename(lora_path)}")
    print(f"[RCLoRAApply] Total keys: {len(all_keys)}  lora_A pairs: {len(lora_a_keys)}  alpha tensors: {len(alpha_keys)}")
    if lora_a_keys:
        sample_key = lora_a_keys[0]
        sample_A = cleaned[sample_key]
        rank = sample_A.shape[0]
        alpha_k = sample_key.replace(".lora_A.weight", ".alpha")
        if alpha_k in cleaned:
            alpha_val = cleaned[alpha_k].item()
            print(f"[RCLoRAApply] rank={rank}  alpha={alpha_val}  alpha/rank={alpha_val/rank:.4f}  effective_scale={scale * alpha_val / rank:.4f}")
        else:
            print(f"[RCLoRAApply] rank={rank}  no alpha tensor found  effective_scale={scale:.4f}")
        print(f"[RCLoRAApply] Sample key: {sample_key}  shape A={sample_A.shape}  B={cleaned[sample_key.replace('.lora_A.','.lora_B.')].shape}")

    applied = 0
    skipped = 0

    for key, val in cleaned.items():
        if ".lora_A.weight" not in key:
            continue

        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in cleaned:
            continue

        lora_A = cleaned[key]   # (rank, in_features)
        lora_B = cleaned[b_key] # (out_features, rank)
        rank = lora_A.shape[0]

        # Apply alpha/rank scaling if stored in the file
        alpha_key = key.replace(".lora_A.weight", ".alpha")
        if alpha_key in cleaned:
            alpha = cleaned[alpha_key].item()
            effective_scale = scale * (alpha / rank)
        else:
            effective_scale = scale

        delta = (lora_B @ lora_A) * effective_scale  # (out_features, in_features)

        # Navigate to the target parameter
        base_key = key.replace(".lora_A.weight", "")
        parts = base_key.split(".")
        module = transformer
        for part in parts[:-1]:
            # Handle ModuleList integer indices
            if part.isdigit():
                try:
                    module = module[int(part)]
                except (IndexError, TypeError):
                    module = None
            else:
                module = getattr(module, part, None)
            if module is None:
                break

        if module is None:
            skipped += 1
            continue

        param = getattr(module, parts[-1], None)
        if param is None:
            skipped += 1
            continue

        try:
            # Direct Parameter (e.g. img_in.weight)
            if hasattr(param, "data"):
                param.data.add_(delta.to(device=param.device, dtype=param.dtype))
                applied += 1
            # nn.Linear or similar — target its .weight
            elif hasattr(param, "weight") and hasattr(param.weight, "data"):
                w = param.weight
                w.data.add_(delta.to(device=w.device, dtype=w.dtype))
                applied += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[RCLoRAApply] Warning: could not apply {base_key}: {e}")
            skipped += 1

    print(f"[RCLoRAApply] Applied {applied} LoRA layers, skipped {skipped}, scale={scale}")
    return applied


class RCLoRAApply:
    """
    Apply a second LoRA to an already-loaded AITK_PIPELINE.
    Connect: RCAITKLoadPipeline → RCLoRAApply → RCAITKGenerate
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe":          ("AITK_PIPELINE",),
                "lora":          ("AITK_LORA",),
            }
        }

    RETURN_TYPES  = ("AITK_PIPELINE",)
    RETURN_NAMES  = ("pipe",)
    FUNCTION      = "apply"
    CATEGORY      = "RunComfy/AITK"
    DESCRIPTION   = "Apply a second LoRA on top of an already-loaded AITK_PIPELINE."

    def apply(self, pipe, lora: dict):
        url_or_path = (lora.get("lora_path_or_url") or "").strip()
        name        = (lora.get("lora_name") or "").strip()
        scale       = float(lora.get("lora_scale", 1.0))

        if not url_or_path and not name:
            print("[RCLoRAApply] No LoRA specified, passing pipe through unchanged.")
            return (pipe,)

        # Resolve path
        if url_or_path:
            lora_path = _download_lora(url_or_path)
        else:
            # Fall back to ComfyUI loras folder
            try:
                import folder_paths
                lora_path = folder_paths.get_full_path_or_raise("loras", name)
            except Exception as e:
                raise ValueError(f"[RCLoRAApply] Cannot resolve LoRA '{name}': {e}")

        transformer = _get_transformer(pipe)
        if transformer is None:
            raise RuntimeError(
                "[RCLoRAApply] Could not find transformer in pipeline object. "
                f"Pipeline type: {type(pipe)}. Attributes: {dir(pipe)}"
            )
        print(f"[RCLoRAApply] Transformer type: {type(transformer).__name__}, "
              f"device: {next(transformer.parameters()).device}, "
              f"dtype: {next(transformer.parameters()).dtype}")

        _apply_lora(transformer, lora_path, scale)
        return (pipe,)


NODE_CLASS_MAPPINGS = {
    "RCLoRAApply": RCLoRAApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RCLoRAApply": "Apply LoRA to Pipeline",
}
