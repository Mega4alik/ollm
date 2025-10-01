from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Sequence, Union

import torch


@dataclass
class DiffusionRunConfig:
    prompt: Optional[Union[str, Sequence[str]]] = None
    negative_prompt: Optional[Union[str, Sequence[str]]] = None
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    height: Optional[int] = None
    width: Optional[int] = None
    generator: Optional[Union[int, torch.Generator]] = None
    output_type: str = "pil"
    eta: Optional[float] = None
    denoising_start: Optional[float] = None
    denoising_end: Optional[float] = None
    strength: Optional[float] = None
    image: Optional[Any] = None
    mask_image: Optional[Any] = None
    control_image: Optional[Any] = None
    controlnet_conditioning_image: Optional[Any] = None
    guidance_rescale: Optional[float] = None
    prompt_embeds: Optional[Any] = None
    negative_prompt_embeds: Optional[Any] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)


class DiffusionRunner:
    """Executes diffusion denoising loops with resource-aware defaults."""

    def __init__(
        self,
        pipeline,
        device: torch.device,
        torch_dtype: torch.dtype = torch.float16,
        scheduler_override: Optional[str] = None,
    ) -> None:
        self.pipeline = pipeline
        self.device = device
        self.torch_dtype = torch_dtype
        self._apply_scheduler_override(scheduler_override)
        self.pipeline.set_progress_bar_config(leave=False)

    def _apply_scheduler_override(self, scheduler_name: Optional[str]) -> None:
        if not scheduler_name:
            return
        from importlib import import_module

        module = import_module("diffusers.schedulers")
        scheduler_cls = getattr(module, scheduler_name, None)
        if scheduler_cls is None:
            raise ValueError(f"Unknown scheduler '{scheduler_name}'")
        self.pipeline.scheduler = scheduler_cls.from_config(self.pipeline.scheduler.config)

    def _prepare_generator(self, generator) -> Optional[torch.Generator]:
        if generator is None:
            return None
        if isinstance(generator, torch.Generator):
            return generator
        if isinstance(generator, int):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(generator)
            return gen
        raise TypeError("generator must be None, an int seed, or a torch.Generator instance")

    def generate(self, config: Optional[DiffusionRunConfig] = None, **kwargs):
        if config is None:
            known_fields = {f.name for f in fields(DiffusionRunConfig)}
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
            extra_options = {k: v for k, v in kwargs.items() if k not in known_fields}
            config = DiffusionRunConfig(**cfg_kwargs)
            if extra_options:
                config.extra_options.update(extra_options)
        elif kwargs:
            config.extra_options.update(kwargs)

        generator = self._prepare_generator(config.generator)
        call_kwargs: Dict[str, Any] = {
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "output_type": config.output_type,
        }

        if config.prompt is not None:
            call_kwargs["prompt"] = config.prompt
        if config.negative_prompt is not None:
            call_kwargs["negative_prompt"] = config.negative_prompt
        if config.height is not None:
            call_kwargs["height"] = config.height
        if config.width is not None:
            call_kwargs["width"] = config.width
        if generator is not None:
            call_kwargs["generator"] = generator
        if config.eta is not None:
            call_kwargs["eta"] = config.eta
        if config.denoising_start is not None:
            call_kwargs["denoising_start"] = config.denoising_start
        if config.denoising_end is not None:
            call_kwargs["denoising_end"] = config.denoising_end
        if config.strength is not None:
            call_kwargs["strength"] = config.strength
        if config.image is not None:
            call_kwargs["image"] = config.image
        if config.mask_image is not None:
            call_kwargs["mask_image"] = config.mask_image
        if config.control_image is not None:
            call_kwargs["control_image"] = config.control_image
        if config.controlnet_conditioning_image is not None:
            call_kwargs["controlnet_conditioning_image"] = config.controlnet_conditioning_image
        if config.guidance_rescale is not None:
            call_kwargs["guidance_rescale"] = config.guidance_rescale
        if config.prompt_embeds is not None:
            call_kwargs["prompt_embeds"] = config.prompt_embeds
        if config.negative_prompt_embeds is not None:
            call_kwargs["negative_prompt_embeds"] = config.negative_prompt_embeds

        if config.extra_options:
            call_kwargs.update(config.extra_options)

        return self.pipeline(**call_kwargs)
