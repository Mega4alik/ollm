from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Type

from .base import PipelineAdapter


@dataclass
class Registry:
    """Simple registry that maps model identifiers to adapter factories."""

    _factories: Dict[str, Callable[..., PipelineAdapter]] = field(default_factory=dict)

    def register(self, model_ids: Iterable[str], factory: Callable[..., PipelineAdapter]) -> None:
        for model_id in model_ids:
            if model_id in self._factories:
                raise ValueError(f"Model id '{model_id}' is already registered")
            self._factories[model_id] = factory

    def get(self, model_id: str) -> Callable[..., PipelineAdapter]:
        try:
            return self._factories[model_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model id '{model_id}'. Available models: {sorted(self._factories.keys())}"
            ) from exc

    def list(self) -> Iterable[str]:
        return self._factories.keys()


registry = Registry()


def register_adapter(model_ids: Iterable[str]):
    """Decorator that registers the decorated adapter class for the given ids."""

    def decorator(cls: Type[PipelineAdapter]):
        registry.register(model_ids, lambda *args, **kwargs: cls(*args, **kwargs))
        return cls

    return decorator
