"""
Dependency injection container using dependency-injector.
"""

from dependency_injector import containers, providers

from .config import Settings
from .libs.storage import InMemoryStorage
from .services.pipeline_manager import PipelineManager
from .tasks.executor import InferenceExecutor


class Container(containers.DeclarativeContainer):
    """Dependency injection container."""

    # Configuration
    config = providers.Singleton(Settings)

    # In-memory storage
    storage = providers.Singleton(
        InMemoryStorage,
        max_size=1000,
    )

    # Pipeline manager (hf_token is passed per request now)
    pipeline_manager = providers.Singleton(
        PipelineManager,
        device=config.provided.device,
        enable_cpu_offload=config.provided.enable_cpu_offload,
        lora_download_cache_dir=config.provided.lora_download_cache_dir,
    )

    # Inference executor
    inference_executor = providers.Singleton(
        InferenceExecutor,
        storage=storage,
        pipeline_manager=pipeline_manager,
        output_base_path=config.provided.output_base_path,
        inference_timeout=config.provided.inference_timeout,
    )
