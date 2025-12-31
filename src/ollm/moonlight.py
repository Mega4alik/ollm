# Moonlight-16B (DeepseekV3 Architecture)

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
from .modeling_moonlight import DeepseekV3ForCausalLM, DeepseekV3Model, DeepseekV3DecoderLayer, DeepseekV3Config, DeepseekV3RMSNorm, DeepseekV3MLP, DeepseekV3MoE
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


class MyDeepseekV3DecoderLayer(DeepseekV3DecoderLayer, loaderLayer):
	def __init__(self, config: DeepseekV3Config, layer_idx: int):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyDeepseekV3Model(DeepseekV3Model):
	def __init__(self, config: DeepseekV3Config):
		super().__init__(config)
        # Re-initialize layers
		self.layers = nn.ModuleList([MyDeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
		# Unload initially
		for layer in self.layers:
			layer._unload_layer_weights()

	def forward(self, *args, **kwargs):
		if "input_ids" in kwargs:
			self.embed_tokens.to(kwargs["input_ids"].device)
		elif len(args) > 0 and args[0] is not None:
			self.embed_tokens.to(args[0].device)

		self.embed_tokens.cpu()
		out = super().forward(*args, **kwargs)
		self.embed_tokens.to(out.last_hidden_state.device)
		return out

# Monkey-patching module
import src.ollm.modeling_moonlight as modeling_moonlight
modeling_moonlight.DeepseekV3DecoderLayer = MyDeepseekV3DecoderLayer
modeling_moonlight.DeepseekV3Model = MyDeepseekV3Model


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


class MyMoonlightForCausalLM(DeepseekV3ForCausalLM, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.model = MyDeepseekV3Model(config)
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)
