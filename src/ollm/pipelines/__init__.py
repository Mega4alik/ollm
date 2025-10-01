"""Pipeline adapters for oLLM inference runtimes."""

from .registry import Registry, registry, register_adapter
from .base import PipelineAdapter

__all__ = ["PipelineAdapter", "Registry", "registry", "register_adapter"]
