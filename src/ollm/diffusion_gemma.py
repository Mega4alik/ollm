# diffusion_gemma
import time, os, math, json
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder, file_get_contents

#global vars
loader, stats = None, None

#======== rewriting core classes ==============
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
	create_sliding_window_causal_mask, create_causal_mask, repeat_kv, TransformersKwargs, Cache, BaseModelOutputWithPast, DiffusionGemmaBlockDiffusionOutputWithPast, 
	DiffusionGemmaForBlockDiffusion, DiffusionGemmaDecoderModel, DiffusionGemmaDecoderTextLayer
)


class loaderLayer:
	def _load_layer_weights(self):
		print("loaderLayer");exit()
		t1 = time.perf_counter()
		base = f"model.decoder.layers.{self.layer_idx}."
		loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)
			
	def _unload_layer_weights(self):
		base = f"model.decoder.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			print("unloading", attr_path)
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)



class MyDiffusionGemmaDecoderTextLayer(DiffusionGemmaDecoderTextLayer, loaderLayer):
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyDiffusionGemmaDecoderModel(DiffusionGemmaDecoderModel):
	def __init__(self, config):
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList()
		for layer_idx in range(config.text_config.num_hidden_layers):
			self.layers.append(MyDiffusionGemmaDecoderTextLayer(config.text_config, layer_idx))
			self.layers[-1]._unload_layer_weights()

	def forward(self, **args):
		out = super().forward(**args)
		if stats: print("./gemma3.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		return out



import transformers.models.diffusion_gemma.modeling_diffusion_gemma as modeling
modeling.DiffusionGemmaDecoderModel = MyDiffusionGemmaDecoderModel

#===============================================================================


class oForGeneration(loaderLayer):
	def generate(self, **args):
		with torch.no_grad():			
			return super().generate(**args)

class MyDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		#self.num_hidden_layers = config.text_config.num_hidden_layers

