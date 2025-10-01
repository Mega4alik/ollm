"""Utilities to adapt Hugging Face diffusion pipelines to tiny GPUs.

The helpers in this module wrap common low-VRAM strategies so that callers do not
need to remember the exact sequence of method calls supported by each
``diffusers`` release.  They intentionally mirror the philosophy used for large
language models inside oLLM: aggressively stream weights from disk/CPU and keep
activations small via tiling/chunking.  While full NVMe streaming for UNet
layers is work in progress, these knobs unlock the built-in optimisations that
``diffusers`` exposes today (sequential offload, attention slicing, VAE tiling,
etc.) and provide a single place to evolve the feature set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class DiffusionOptimizationConfig:
    """Toggles for making large diffusion checkpoints run on small GPUs."""

    sequential_cpu_offload: bool = True
    model_cpu_offload: bool = False
    attention_slicing: Optional[Any] = "auto"
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = False
    forward_chunk_size: Optional[int] = 2
    enable_xformers: bool = False
    enable_channels_last: bool = True
    text_encoder_offload: str = "cpu"
    compile_unet: bool = False
    max_attention_window: Optional[int] = None


DEFAULT_OPTIMIZATIONS = DiffusionOptimizationConfig()


def build_optimizations(**overrides: Any) -> DiffusionOptimizationConfig:
    """Return a config object with user overrides applied safely."""

    valid_fields = DiffusionOptimizationConfig.__annotations__.keys()
    filtered = {k: v for k, v in overrides.items() if k in valid_fields and v is not None}
    if not filtered:
        return DEFAULT_OPTIMIZATIONS
    return replace(DEFAULT_OPTIMIZATIONS, **filtered)


def apply_diffusion_optimizations(pipeline, config: DiffusionOptimizationConfig, device: torch.device) -> None:
    """Apply the requested low-VRAM strategies to a ``diffusers`` pipeline."""

    if config.enable_channels_last and hasattr(pipeline, "unet"):
        try:
            pipeline.unet.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if config.enable_xformers and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    if config.attention_slicing is not False and hasattr(pipeline, "enable_attention_slicing"):
        try:
            if config.attention_slicing in (True, "auto"):
                pipeline.enable_attention_slicing()
            else:
                pipeline.enable_attention_slicing(config.attention_slicing)
        except Exception:
            pass

    if config.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
        try:
            pipeline.enable_vae_tiling()
        except Exception:
            pass

    if config.enable_vae_slicing and hasattr(getattr(pipeline, "vae", None), "enable_slicing"):
        try:
            pipeline.vae.enable_slicing()
        except Exception:
            pass

    if config.forward_chunk_size and hasattr(getattr(pipeline, "unet", None), "enable_forward_chunking"):
        try:
            pipeline.unet.enable_forward_chunking(config.forward_chunk_size)
        except Exception:
            pass

    if config.max_attention_window and hasattr(getattr(pipeline, "unet", None), "set_attention_slice"):
        try:
            pipeline.unet.set_attention_slice(config.max_attention_window)
        except Exception:
            pass

    # CPU/GPU offload hierarchy mirrors diffusers utilities.  Sequential CPU
    # offload keeps only the working module on GPU, which mirrors the oLLM
    # philosophy most closely.
    if config.sequential_cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
        try:
            pipeline.enable_sequential_cpu_offload(device)
            offloaded = True
        except Exception:
            offloaded = False
    else:
        offloaded = False

    if not offloaded and config.model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
        try:
            pipeline.enable_model_cpu_offload(device)
            offloaded = True
        except Exception:
            offloaded = False

    # Text encoders are large and only needed during prompt encoding.  To avoid
    # them occupying scarce VRAM, move them to CPU unless explicitly requested
    # otherwise.
    if config.text_encoder_offload != "gpu" and hasattr(pipeline, "text_encoder"):
        try:
            target = "cpu" if config.text_encoder_offload == "cpu" else config.text_encoder_offload
            pipeline.text_encoder.to(target)
        except Exception:
            pass

    if config.compile_unet and hasattr(torch, "compile") and hasattr(pipeline, "unet"):
        try:
            pipeline.unet = torch.compile(pipeline.unet)
        except Exception:
            pass

    pipeline.set_progress_bar_config(disable=None)
