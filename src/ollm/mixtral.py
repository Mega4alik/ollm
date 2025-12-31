# Falcon3-MoE-2x7B (Mixtral Architecture)

import time, os
from datetime import datetime
import threading
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder

# shared objects
loader, stats = None, None

#======== rewriting core classes ==============
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralAttention, MixtralSparseMoeBlock, MixtralDecoderLayer, MixtralModel, MixtralConfig, create_causal_mask, Cache
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    from transformers.models.mixtral.modeling_mixtral import DynamicCache

from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast

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


class MyMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    # Standard forward, but weights are loaded via loaderLayer in the DecoderLayer wrapper
    pass

class MyMixtralDecoderLayer(MixtralDecoderLayer, loaderLayer):
	def __init__(self, config: MixtralConfig, layer_idx: int):
		self.layer_idx = layer_idx
		super().__init__(config, layer_idx)

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyMixtralModel(MixtralModel):
	def __init__(self, config):
		super().__init__(config)
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyMixtralDecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		**kwargs: Any,
	) -> MoeModelOutputWithPast:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if inputs_embeds is None:
			self.embed_tokens.to(input_ids.device)
			inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

		if use_cache and past_key_values is None:
			past_key_values = DynamicCache()

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position: torch.Tensor = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)

		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		causal_mask = create_causal_mask(
			config=self.config,
			input_embeds=inputs_embeds,
			attention_mask=attention_mask,
			cache_position=cache_position,
			past_key_values=past_key_values,
			position_ids=position_ids,
		)

		hidden_states = inputs_embeds
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		#============= meine ==============
		self.embed_tokens.cpu(); self.parent_lm_head.cpu()

		for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask=causal_mask,
				position_ids=position_ids,
				past_key_value=past_key_values,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)

		hidden_states = self.norm(hidden_states)
		self.embed_tokens.to(hidden_states.device); self.parent_lm_head.to(hidden_states.device)
		if stats: print("./Mixtral.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		#====================================

		return MoeModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)

# Monkey-patch
import transformers.models.mixtral.modeling_mixtral as mixtral_modeling
mixtral_modeling.MixtralDecoderLayer = MyMixtralDecoderLayer
mixtral_modeling.MixtralModel = MyMixtralModel
#===============================================

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


class MyMixtralForCausalLM(MixtralForCausalLM, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)

	def offload_layers_to_cpu_gds(self, layers_num=2):
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			for name, attr in loader.manifest.items():
				if name.startswith(f"model.layers.{layer_idx}."):
					loader.offload_param_to_cpu(name)
		print("./Mixtral offloading layers to CPU. Done.")
