from __future__ import annotations

import inspect
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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


@dataclass(frozen=True)
class _PromptCacheKey:
    prompt: Tuple[str, ...]
    negative: Tuple[str, ...]
    guidance: float
    dtype: torch.dtype
    device: str
    num_images_per_prompt: int
    clip_skip: Optional[int]


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
        self._prompt_cache: "OrderedDict[_PromptCacheKey, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]" = OrderedDict()
        self._max_cache_entries = 8

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

        do_guidance = config.guidance_scale is not None and config.guidance_scale > 1.0
        num_images_per_prompt = int(config.extra_options.pop("num_images_per_prompt", 1) or 1)
        clip_skip = config.extra_options.pop("clip_skip", None)

        if num_images_per_prompt > 1:
            call_kwargs["num_images_per_prompt"] = num_images_per_prompt
        if clip_skip is not None:
            call_kwargs["clip_skip"] = clip_skip

        prompt_embeds, negative_embeds = self._resolve_prompt_embeddings(
            config,
            do_guidance,
            num_images_per_prompt,
            clip_skip=clip_skip,
        )

        if prompt_embeds is not None:
            call_kwargs["prompt_embeds"] = prompt_embeds
            call_kwargs.pop("prompt", None)
        elif config.prompt is not None:
            call_kwargs["prompt"] = config.prompt

        if negative_embeds is not None:
            call_kwargs["negative_prompt_embeds"] = negative_embeds
            call_kwargs.pop("negative_prompt", None)
        elif config.negative_prompt is not None:
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

    # ------------------------------------------------------------------
    # Prompt embedding cache + helpers
    # ------------------------------------------------------------------

    def clear_prompt_cache(self) -> None:
        """Drop all cached prompt embeddings."""

        self._prompt_cache.clear()

    def _resolve_prompt_embeddings(
        self,
        config: DiffusionRunConfig,
        do_guidance: bool,
        num_images_per_prompt: int,
        clip_skip: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if config.prompt_embeds is not None:
            prompt = config.prompt_embeds.to(self.device)
            negative = (
                config.negative_prompt_embeds.to(self.device)
                if config.negative_prompt_embeds is not None
                else None
            )
            return prompt, negative

        encode_fn = getattr(self.pipeline, "_encode_prompt", None)
        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if encode_fn is None or tokenizer is None:
            return None, None

        prompt_tuple = self._normalize_prompt(config.prompt)
        negative_tuple = self._normalize_prompt(config.negative_prompt)
        cache_key = _PromptCacheKey(
            prompt=prompt_tuple,
            negative=negative_tuple,
            guidance=float(config.guidance_scale or 0.0),
            dtype=self.torch_dtype,
            device=str(self.device),
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
        )

        if cache_key in self._prompt_cache:
            prompt_embeds, negative_embeds = self._prompt_cache[cache_key]
            # Move to end to maintain LRU semantics
            self._prompt_cache.move_to_end(cache_key)
            return (
                prompt_embeds.to(self.device) if prompt_embeds is not None else None,
                negative_embeds.to(self.device) if negative_embeds is not None else None,
            )

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt_tuple if len(prompt_tuple) > 1 else prompt_tuple[0] if prompt_tuple else None,
            "device": self.device,
            "num_images_per_prompt": num_images_per_prompt,
            "do_classifier_free_guidance": do_guidance,
            "negative_prompt": negative_tuple if len(negative_tuple) > 1 else (
                negative_tuple[0] if negative_tuple else None
            ),
            "clip_skip": clip_skip,
        }
        call_kwargs = self._filter_kwargs(encode_fn, call_kwargs)

        outputs = encode_fn(**call_kwargs)
        if isinstance(outputs, tuple):
            prompt_embeds = outputs[0]
            negative_embeds = outputs[1] if len(outputs) > 1 else None
        else:
            prompt_embeds = outputs
            negative_embeds = None

        prompt_embeds = prompt_embeds.to(self.device)
        if negative_embeds is not None:
            negative_embeds = negative_embeds.to(self.device)

        self._store_prompt_cache(cache_key, prompt_embeds, negative_embeds)
        return prompt_embeds, negative_embeds

    def _store_prompt_cache(
        self,
        key: _PromptCacheKey,
        prompt_embeds: Optional[torch.Tensor],
        negative_embeds: Optional[torch.Tensor],
    ) -> None:
        if key in self._prompt_cache:
            self._prompt_cache.move_to_end(key)
        self._prompt_cache[key] = (
            prompt_embeds.detach().to("cpu") if prompt_embeds is not None else None,
            negative_embeds.detach().to("cpu") if negative_embeds is not None else None,
        )
        while len(self._prompt_cache) > self._max_cache_entries:
            self._prompt_cache.popitem(last=False)

    @staticmethod
    def _normalize_prompt(value: Optional[Union[str, Sequence[str]]]) -> Tuple[str, ...]:
        if value is None:
            return tuple()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    @staticmethod
    def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
