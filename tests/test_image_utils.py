"""
Tests for image utilities.
"""

import base64
import io
import pytest
from PIL import Image

from src.libs.image_utils import (
    load_image_from_source,
    encode_image_to_base64,
)


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def sample_base64(sample_image):
    """Create base64 encoded image data."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestLoadImageFromSource:
    """Tests for load_image_from_source."""

    def test_load_from_base64(self, sample_base64):
        """Should load image from base64 string."""
        image = load_image_from_source(base64_data=sample_base64)
        assert image is not None
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    def test_load_from_data_url(self, sample_base64):
        """Should load image from data URL format."""
        data_url = f"data:image/png;base64,{sample_base64}"
        image = load_image_from_source(base64_data=data_url)
        assert image is not None
        assert isinstance(image, Image.Image)

    def test_load_none_returns_none(self):
        """Should return None when no source provided."""
        assert load_image_from_source() is None

    def test_invalid_base64_raises(self):
        """Should raise ValueError for invalid base64."""
        with pytest.raises(ValueError):
            load_image_from_source(base64_data="not-valid-base64!")

    def test_nonexistent_file_raises(self):
        """Should raise ValueError for nonexistent file."""
        with pytest.raises(ValueError):
            load_image_from_source(local_path="/nonexistent/file.jpg")


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64."""

    def test_encode_jpeg(self, sample_image):
        """Should encode image as JPEG."""
        encoded = encode_image_to_base64(sample_image, format="JPEG")
        assert isinstance(encoded, str)

        # Decode and verify
        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "JPEG"

    def test_encode_png(self, sample_image):
        """Should encode image as PNG."""
        encoded = encode_image_to_base64(sample_image, format="PNG")
        assert isinstance(encoded, str)

        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "PNG"

    def test_encode_webp(self, sample_image):
        """Should encode image as WEBP."""
        encoded = encode_image_to_base64(sample_image, format="WEBP")
        assert isinstance(encoded, str)

        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "WEBP"

    def test_encode_rgba_as_jpeg(self):
        """Should convert RGBA to RGB for JPEG."""
        rgba_image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        encoded = encode_image_to_base64(rgba_image, format="JPEG")

        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        assert img.mode == "RGB"
