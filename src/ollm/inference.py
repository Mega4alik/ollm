import os
from typing import Optional

import torch

from .pipelines import registry
from .pipelines import llm_adapter  # noqa: F401
from .pipelines.diffusion import adapter as diffusion_adapter  # noqa: F401
from .utils import Stats


class Inference:
    def __init__(self, model_id, device="cuda:0", logging=True, multimodality=False, **kwargs):
        self.model_id = model_id
        self.device = torch.device(device)
        self.multimodality = multimodality
        self.stats = Stats() if logging else None
        factory = registry.get(model_id)
        self.adapter = factory(
            model_id=model_id,
            device=self.device,
            stats=self.stats,
            multimodality=multimodality,
            **kwargs,
        )
        self.model = None
        self.tokenizer: Optional[object] = None
        self.processor: Optional[object] = None

    def ini_model(self, models_dir="./models/", force_download=False):
        os.makedirs(models_dir, exist_ok=True)
        model_dir = self.adapter.prepare(models_dir, force_download=force_download)
        self.adapter.load(model_dir)
        self.model = self.adapter.model
        self.tokenizer = getattr(self.adapter, "tokenizer", None)
        self.processor = getattr(self.adapter, "processor", None)
        return self.model

    def offload_layers_to_cpu(self, **args):
        if hasattr(self.adapter, "offload_layers_to_cpu"):
            return self.adapter.offload_layers_to_cpu(**args)
        raise AttributeError("Adapter does not support CPU offload")

    def offload_layers_to_gpu_cpu(self, **args):
        if hasattr(self.adapter, "offload_layers_to_gpu_cpu"):
            return self.adapter.offload_layers_to_gpu_cpu(**args)
        raise AttributeError("Adapter does not support GPU/CPU offload")

    def DiskCache(self, cache_dir="./kvcache"):
        if hasattr(self.adapter, "disk_cache"):
            return self.adapter.disk_cache(cache_dir=cache_dir)
        raise AttributeError("Adapter does not implement disk caching")

    def generate(self, *args, **kwargs):
        return self.adapter.generate(*args, **kwargs)
