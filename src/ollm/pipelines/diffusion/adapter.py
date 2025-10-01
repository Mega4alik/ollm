from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, replace
from typing import Dict, Optional, TYPE_CHECKING

import torch

from ..base import PipelineAdapter
from ..registry import register_adapter
from .runner import DiffusionRunner

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from diffusers import DiffusionPipeline


@dataclass
class DiffusionModelConfig:
    model_ids: tuple
    repo_id: Optional[str] = None
    download_url: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    torch_dtype: torch.dtype = torch.float16
    enable_cpu_offload: bool = True
    enable_sequential_offload: bool = True
    enable_vae_tiling: bool = True
    enable_attention_slicing: bool = True
    scheduler_override: Optional[str] = None


_DIFFUSION_MODELS: Dict[str, DiffusionModelConfig] = {
    "sdxl-base-1.0": DiffusionModelConfig(
        model_ids=("sdxl-base-1.0",),
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
    ),
    "qwen-image-edit": DiffusionModelConfig(
        model_ids=("qwen-image-edit", "Qwen/Qwen-Image-Edit-2509"),
        repo_id="Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.float16,
    ),
}


def _maybe_download_zip(url: str, destination_dir: str) -> None:
    if not url:
        raise ValueError(
            "No download URL configured. Set OLLMDIFF_QWEN_IMAGE_EDIT_URL to a direct download for the diffusers weights."
        )
    os.makedirs(destination_dir, exist_ok=True)
    filename = url.split("/")[-1] or "weights.zip"
    zip_path = os.path.join(destination_dir, filename)
    print(f"Downloading diffusion weights from {url} ...")

    import requests

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {zip_path}")
    print("Unpacking archive ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    os.remove(zip_path)


@register_adapter([model_id for config in _DIFFUSION_MODELS.values() for model_id in config.model_ids])
class DiffusionPipelineAdapter(PipelineAdapter):
    """Adapter that loads diffusion pipelines with aggressive offloading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested_model_id = self.model_id
        self.config = self._resolve_config()
        if self.requested_model_id != self.cache_id:
            print(
                f"Redirecting requested diffusion id '{self.requested_model_id}' to registered '{self.cache_id}'"
            )
        overrides = {}
        for field in (
            "download_url",
            "repo_id",
            "revision",
            "variant",
            "torch_dtype",
            "enable_cpu_offload",
            "enable_sequential_offload",
            "enable_vae_tiling",
            "enable_attention_slicing",
            "scheduler_override",
        ):
            if field in self.kwargs and self.kwargs[field] is not None:
                overrides[field] = self.kwargs[field]
        if overrides:
            self.config = replace(self.config, **overrides)
        self.runner: Optional[DiffusionRunner] = None

    def _resolve_config(self) -> DiffusionModelConfig:
        if self.model_id in _DIFFUSION_MODELS:
            self.cache_id = self.model_id
            return _DIFFUSION_MODELS[self.model_id]

        for cache_id, config in _DIFFUSION_MODELS.items():
            if self.model_id in config.model_ids:
                self.cache_id = cache_id
                return config

        raise KeyError(f"Unknown diffusion model '{self.model_id}'")

    def prepare(self, models_dir: str, force_download: bool = False) -> str:
        os.makedirs(models_dir, exist_ok=True)
        model_dir = os.path.join(models_dir, self.cache_id)
        if os.path.exists(model_dir) and not force_download:
            return model_dir

        if os.path.exists(model_dir) and force_download:
            print(f"Removing existing model directory {model_dir} for fresh download")
            import shutil

            shutil.rmtree(model_dir)

        sanitized = self.cache_id.upper().replace("-", "_").replace("/", "_")

        if self.config.repo_id:
            from huggingface_hub import snapshot_download

            print(f"Downloading {self.config.repo_id} ...")
            snapshot_download(
                repo_id=self.config.repo_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                revision=self.config.revision,
            )
        else:
            download_url = self.config.download_url
            if download_url is None:
                env_key = f"OLLMDIFF_{sanitized}_URL"
                download_url = os.environ.get(env_key)
            if download_url:
                _maybe_download_zip(download_url, model_dir)
            else:
                raise ValueError(
                    f"Model '{self.requested_model_id}' has no repository or download URL configured. "
                    "Provide download_url=... or set the environment variable "
                    f"OLLMDIFF_{sanitized}_URL"
                )

        return model_dir

    def load(self, model_dir: str) -> None:
        from diffusers import DiffusionPipeline

        print(f"Loading diffusion pipeline from {model_dir}")
        pipeline = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=True,
            variant=self.config.variant,
        )

        if self.config.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        if self.config.enable_vae_tiling and hasattr(pipeline, "vae"):
            pipeline.enable_vae_tiling()

        if self.config.enable_sequential_offload:
            pipeline.enable_sequential_cpu_offload(self.device)
        elif self.config.enable_cpu_offload:
            pipeline.enable_model_cpu_offload(self.device)

        try:
            pipeline.unet.to(memory_format=torch.channels_last)
        except Exception:
            pass

        self.model = pipeline
        self.runner = DiffusionRunner(
            pipeline=pipeline,
            device=self.device,
            torch_dtype=self.config.torch_dtype,
            scheduler_override=self.config.scheduler_override,
        )

    def generate(self, *args, **kwargs):
        if self.runner is None:
            raise RuntimeError("Pipeline is not loaded")
        return self.runner.generate(*args, **kwargs)

    def metadata(self) -> Dict[str, str]:
        return {"type": "diffusion"}
