"""
Tests for request and response schemas.
"""

import pytest
from pydantic import ValidationError

from src.schemas.request import InferenceInput, LoraItem, PromptItem
from src.schemas.response import (
    InferenceResponse,
    InferenceStatusResponse,
    InferenceResultResponse,
)
from src.schemas.task import InferenceTask, RequestStatus


class TestPromptItem:
    """Tests for PromptItem schema."""

    def test_basic_prompt(self):
        """Basic prompt should work."""
        item = PromptItem(prompt="Hello world")
        assert item.prompt == "Hello world"
        assert item.width is None
        assert item.height is None

    def test_prompt_with_parameters(self):
        """Prompt with parameters should work."""
        item = PromptItem(
            prompt="Hello", width=1024, height=768, seed=42, guidance_scale=4.0, sample_steps=25, neg="bad quality"
        )
        assert item.width == 1024
        assert item.height == 768
        assert item.seed == 42
        assert item.guidance_scale == 4.0
        assert item.sample_steps == 25
        assert item.neg == "bad quality"

    def test_prompt_with_control_image(self):
        """Prompt with control image should work."""
        item = PromptItem(prompt="Hello", ctrl_img="base64data")
        assert item.ctrl_img == "base64data"
        assert item.get_control_image() == "base64data"

    def test_prompt_with_multiple_control_images(self):
        """Prompt with multiple control images should work."""
        item = PromptItem(prompt="Hello", ctrl_img_1="base64data1", ctrl_img_2="base64data2", ctrl_img_3="base64data3")
        images = item.get_control_images()
        assert len(images) == 3
        assert images[0] == "base64data1"
        assert images[1] == "base64data2"
        assert images[2] == "base64data3"

    def test_get_control_images_fallback(self):
        """get_control_images should fallback to ctrl_img if no numbered images."""
        item = PromptItem(prompt="Hello", ctrl_img="base64data")
        images = item.get_control_images()
        assert len(images) == 1
        assert images[0] == "base64data"


class TestInferenceInput:
    """Tests for InferenceInput schema."""

    def test_basic_request(self):
        """Basic request should work."""
        req = InferenceInput(
            model="flux",
            loras=[LoraItem(path="my_lora.safetensors")],
            prompts=[PromptItem(prompt="Generate an image")],
        )
        assert req.model == "flux"
        assert len(req.loras) == 1
        assert len(req.prompts) == 1

    def test_request_with_hf_token(self):
        """Request with hf_token should work."""
        req = InferenceInput(
            model="flux",
            loras=[LoraItem(path="my_lora.safetensors")],
            prompts=[PromptItem(prompt="Generate an image")],
            hf_token="hf_xxx",
        )
        assert req.hf_token == "hf_xxx"

    def test_request_with_trigger_word(self):
        """Request with trigger_word should work."""
        req = InferenceInput(
            model="flux",
            loras=[LoraItem(path="my_lora.safetensors")],
            prompts=[PromptItem(prompt="[trigger] a woman reading a book")],
            trigger_word="lilyxyz",
        )
        assert req.trigger_word == "lilyxyz"

    def test_request_with_multiple_prompts(self):
        """Request with multiple prompts should work."""
        req = InferenceInput(
            model="flux",
            loras=[LoraItem(path="my_lora.safetensors")],
            prompts=[
                PromptItem(prompt="First", width=1024, height=1024),
                PromptItem(prompt="Second", width=512, height=512),
            ],
        )
        assert len(req.get_prompts()) == 2
        assert req.prompts[0].width == 1024
        assert req.prompts[1].width == 512

    def test_must_have_prompts(self):
        """Must provide prompts."""
        with pytest.raises(ValidationError):
            InferenceInput(
                model="flux",
                loras=[LoraItem(path="my_lora.safetensors")],
                prompts=[],  # Empty list should fail
            )

    def test_must_have_loras(self):
        """Must provide loras."""
        with pytest.raises(ValidationError):
            InferenceInput(
                model="flux",
                # Missing loras
                prompts=[PromptItem(prompt="Test")],
            )

    def test_get_prompts(self):
        """get_prompts should return prompts list."""
        req = InferenceInput(
            model="flux",
            loras=[LoraItem(path="my_lora.safetensors")],
            prompts=[
                PromptItem(prompt="A"),
                PromptItem(prompt="B"),
            ],
        )
        prompts = req.get_prompts()
        assert len(prompts) == 2
        assert prompts[0].prompt == "A"
        assert prompts[1].prompt == "B"


class TestInferenceTask:
    """Tests for InferenceTask schema."""

    def test_task_creation(self):
        """Task creation should work."""
        task = InferenceTask(
            id="test-id",
            model="flux",
            lora_path_name="my_lora",
            lora_paths=["/path/to/lora.safetensors"],
        )
        assert task.id == "test-id"
        assert task.status == RequestStatus.QUEUED

    def test_mark_as_processing(self):
        """mark_as_processing should update status."""
        task = InferenceTask(
            id="test-id",
            model="flux",
            lora_path_name="my_lora",
            lora_paths=["/path/to/lora.safetensors"],
        )
        task.mark_as_processing()
        assert task.status == RequestStatus.PROCESSING
        assert task.started_at is not None

    def test_mark_as_succeeded(self):
        """mark_as_succeeded should update status and outputs."""
        task = InferenceTask(
            id="test-id",
            model="flux",
            lora_path_name="my_lora",
            lora_paths=["/path/to/lora.safetensors"],
        )
        task.mark_as_processing()
        task.mark_as_succeeded({"images": []})
        assert task.status == RequestStatus.SUCCEEDED
        assert task.finished_at is not None
        assert task.outputs == {"images": []}

    def test_mark_as_failed(self):
        """mark_as_failed should update status and error."""
        task = InferenceTask(
            id="test-id",
            model="flux",
            lora_path_name="my_lora",
            lora_paths=["/path/to/lora.safetensors"],
        )
        task.mark_as_failed("Something went wrong")
        assert task.status == RequestStatus.FAILED
        assert task.error == "Something went wrong"

    def test_status_to_api_status(self):
        """to_api_status should convert correctly."""
        assert RequestStatus.QUEUED.to_api_status() == "in_queue"
        assert RequestStatus.PROCESSING.to_api_status() == "in_progress"
        assert RequestStatus.SUCCEEDED.to_api_status() == "succeeded"
        assert RequestStatus.FAILED.to_api_status() == "failed"
        assert RequestStatus.CANCELLED.to_api_status() == "canceled"
