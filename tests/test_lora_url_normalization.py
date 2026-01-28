from src.libs.url_utils import normalize_huggingface_file_url, looks_like_html_file


def test_normalize_huggingface_blob_to_resolve():
    url = "https://huggingface.co/ostris/z_image_turbo_childrens_drawings/blob/main/z_image_turbo_childrens_drawings.safetensors"
    normalized, note = normalize_huggingface_file_url(url)
    assert note == "huggingface_blob_to_resolve"
    assert "/blob/" not in normalized
    assert "/resolve/" in normalized
    assert normalized.endswith("/z_image_turbo_childrens_drawings.safetensors")


def test_normalize_huggingface_resolve_unchanged():
    url = "https://huggingface.co/ostris/z_image_turbo_childrens_drawings/resolve/main/z_image_turbo_childrens_drawings.safetensors"
    normalized, note = normalize_huggingface_file_url(url)
    assert normalized == url
    assert note is None


def test_normalize_non_huggingface_unchanged():
    url = "https://example.com/some_lora.safetensors"
    normalized, note = normalize_huggingface_file_url(url)
    assert normalized == url
    assert note is None


def test_huggingface_tree_url_is_flagged():
    url = "https://huggingface.co/ostris/z_image_turbo_childrens_drawings/tree/main"
    normalized, note = normalize_huggingface_file_url(url)
    assert normalized == url
    assert note == "huggingface_tree_url"


def test_looks_like_html_file(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"<!DOCTYPE html><html><head><title>HF</title></head><body>nope</body></html>")
    assert looks_like_html_file(str(p)) is True


def test_does_not_flag_binary_file(tmp_path):
    p = tmp_path / "ok.safetensors"
    # Not a real safetensors file, but importantly not HTML.
    p.write_bytes(b"\x00\x01\x02\x03\x04\x05binary-ish")
    assert looks_like_html_file(str(p)) is False
