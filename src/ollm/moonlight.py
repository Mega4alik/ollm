# Moonlight-16B (DeepseekV3 Architecture)

import time, os
from datetime import datetime
import torch
from torch import nn
from typing import Optional, Any
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder

# shared objects
loader, stats = None, None

# Uses remote code (trust_remote_code=True)
from transformers import AutoModelForCausalLM

class loaderLayer:
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"model.layers.{self.layer_idx}."
		loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)

	def _unload_layer_weights(self):
		base = f"model.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


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

class MyMoonlightForCausalLM:
    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        # Load the original model with trust_remote_code=True
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, **kwargs)

        # Create a dynamic class that inherits from the model's actual class and our mixin
        RemoteClass = model.__class__
        class DynamicMoonlightWrapper(RemoteClass, oForGeneration):
            pass

        # Change the object's class to the new dynamic wrapper
        model.__class__ = DynamicMoonlightWrapper

        # Monkey-patch or wrap layers to support offloading

        def patch_layer(layer, layer_idx):
            layer.layer_idx = layer_idx
            original_forward = layer.forward

            loader_instance = loaderLayer()
            loader_instance.layer_idx = layer_idx

            def new_forward(*args, **kwargs):
                loader_instance._load_layer_weights()
                out = original_forward(*args, **kwargs)
                loader_instance._unload_layer_weights()
                return out

            layer.forward = new_forward
            loader_instance._unload_layer_weights()

        if hasattr(model, "model") and hasattr(model.model, "layers"):
             for i, layer in enumerate(model.model.layers):
                 patch_layer(layer, i)

        model.num_hidden_layers = len(model.model.layers)

        return model
