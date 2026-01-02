# DeepSeek-MoE-16B

import time, os
from datetime import datetime
import threading
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder

# shared objects
loader, stats = None, None

# Import from local modeling file
from .modeling_deepseek import DeepseekForCausalLM, DeepseekModel, DeepseekDecoderLayer, DeepseekConfig, DeepseekRMSNorm, DeepseekMLP, DeepseekMoE
from transformers.modeling_outputs import BaseModelOutputWithPast

class loaderLayer:
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"model.layers.{self.layer_idx}."
		loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			if hasattr(parent, "base_layer"): parent = parent.base_layer #peft lora
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)

	def _unload_layer_weights(self):
		base = f"model.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			if hasattr(parent, "base_layer"): parent = parent.base_layer #peft lora
			_set_meta_placeholder(parent, leaf)


class MyDeepseekDecoderLayer(DeepseekDecoderLayer, loaderLayer):
	def __init__(self, config: DeepseekConfig, layer_idx: int):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyDeepseekModel(DeepseekModel):
	def __init__(self, config: DeepseekConfig):
		super().__init__(config)
        # Re-initialize layers with our custom class
		self.layers = nn.ModuleList([MyDeepseekDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
		self._init_weights(self.layers) # Re-init weights? No, we load them.
        # Unload initially
		for layer in self.layers:
			layer._unload_layer_weights()

	def forward(self, *args, **kwargs):
        # Ensure embeddings on device if needed (DeepSeek might handle it, but for safety)
		if "input_ids" in kwargs:
			self.embed_tokens.to(kwargs["input_ids"].device)
		elif len(args) > 0 and args[0] is not None: # input_ids
			self.embed_tokens.to(args[0].device)

		out = super().forward(*args, **kwargs)
		self.embed_tokens.cpu()
		return out

# Monkey-patching module
from . import modeling_deepseek
modeling_deepseek.DeepseekDecoderLayer = MyDeepseekDecoderLayer
modeling_deepseek.DeepseekModel = MyDeepseekModel


class oForGeneration:
	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			base = f"model.layers.{layer_idx}."
			loader.preload_layer_safetensors(base)
			loader.offload_dict_to_gpu_cpu(base, gpu=False)
		print(f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}")


class MyDeepseekForCausalLM(DeepseekForCausalLM, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.model = MyDeepseekModel(config) # Ensure we use our model
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)
