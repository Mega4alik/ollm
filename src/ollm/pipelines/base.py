from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PipelineAdapter(ABC):
    """Base class for all pipeline adapters used by :class:`~ollm.inference.Inference`.

    Adapters encapsulate model-specific download, loading, and execution logic so
    that the core runtime can support heterogeneous architectures (LLMs, diffusion
    pipelines, etc.).
    """

    def __init__(self, model_id: str, device, stats=None, **kwargs):
        self.model_id = model_id
        self.device = device
        self.stats = stats
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.processor = None

    @abstractmethod
    def prepare(self, models_dir: str, force_download: bool = False) -> str:
        """Ensure the weights/artifacts for the model are available locally.

        Returns the directory that contains the (possibly downloaded) weights.
        """

    @abstractmethod
    def load(self, model_dir: str) -> None:
        """Load the model artifacts into memory."""

    def offload_layers_to_cpu(self, **args):  # pragma: no cover - optional override
        raise NotImplementedError("CPU offload is not implemented for this adapter")

    def offload_layers_to_gpu_cpu(self, **args):  # pragma: no cover - optional override
        raise NotImplementedError("GPU/CPU tiered offload is not implemented for this adapter")

    def disk_cache(self, **args):  # pragma: no cover - optional override
        raise NotImplementedError("Disk cache is not implemented for this adapter")

    def generate(self, *args, **kwargs):  # pragma: no cover - optional override
        raise NotImplementedError("Generation is not implemented for this adapter")

    def metadata(self) -> Optional[Dict[str, Any]]:  # pragma: no cover - optional override
        return None
