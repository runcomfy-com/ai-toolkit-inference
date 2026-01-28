from __future__ import annotations

from typing import Optional, Tuple
from urllib.parse import urlparse


def normalize_huggingface_file_url(url: str) -> Tuple[str, Optional[str]]:
    """
    Normalize common Hugging Face "web UI" URLs into direct-download URLs.

    Users often paste URLs containing `/blob/` which are HTML pages. Downloading those
    produces an HTML file with a `.safetensors` extension and later fails during
    LoRA weight loading.
    """
    if not isinstance(url, str) or not url:
        return url, None

    try:
        parsed = urlparse(url)
    except Exception:
        return url, None

    host = (parsed.netloc or "").lower()
    if host not in ("huggingface.co", "www.huggingface.co"):
        return url, None

    if "/blob/" in parsed.path:
        return url.replace("/blob/", "/resolve/", 1), "huggingface_blob_to_resolve"

    if "/tree/" in parsed.path:
        return url, "huggingface_tree_url"

    return url, None


def looks_like_html_file(path: str) -> bool:
    """Heuristic to detect when a downloaded file is actually HTML."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except Exception:
        return False

    head_l = head.lstrip().lower()
    return (
        head_l.startswith(b"<!doctype html")
        or head_l.startswith(b"<html")
        or b"<head" in head_l[:1024]
        or b"<title" in head_l[:1024]
    )
