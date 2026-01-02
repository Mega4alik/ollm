import os, time, shutil
import torch
from transformers import DynamicCache
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List

class oCache:	
	def ini_ocache(self, cache_dir, device, stats):
		if not cache_dir: raise Error("cache_dir can not be empty. If you are trying to not use DiskCache, simply set past_key_values=None. This will use default DynamicCache")
		self.cache_folder = os.path.join(cache_dir, "kv_cache")
		self.key_cache2, self.value_cache2 = [], []
		if os.path.exists(self.cache_folder): shutil.rmtree(self.cache_folder)
		os.makedirs(self.cache_folder)
		self.device = device
		self.stats = stats

	def load_from_disk(self, layer_idx):
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		if not os.path.exists(path): return None
		t1 = time.perf_counter()
		tensors = torch.load(path, map_location=self.device)
		if self.stats: self.stats.set("kvload", t1)
		return tensors

	def save_to_disk(self, tensors, layer_idx):
		t1 = time.perf_counter()
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		tensors = (tensors[0].cpu(), tensors[1].cpu())
		torch.save(tensors, path)
		if self.stats: self.stats.set("kvsave", t1)


class KVCache(DynamicCache, oCache): #DiskCache
	def __init__(self, cache_dir="./kv_cache", device="cuda:0", stats=None):
		super().__init__()		
		self.ini_ocache(cache_dir, device, stats)

	@property
	def seen_tokens(self):
		return self.get_seq_length()

	def get_max_length(self) -> Optional[int]:
		return None

	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
		cache_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		tensors = self.load_from_disk(layer_idx)
		if tensors is not None:
			self.layers[layer_idx].keys, self.layers[layer_idx].values = tensors
			if layer_idx < len(self.key_cache2):
				self.layers[layer_idx].keys = torch.cat([self.layers[layer_idx].keys, self.key_cache2[layer_idx]], dim=-2)
				self.layers[layer_idx].values = torch.cat([self.layers[layer_idx].values, self.value_cache2[layer_idx]], dim=-2)
				self.key_cache2[layer_idx] = torch.cat([self.key_cache2[layer_idx], key_states], dim=-2)
				self.value_cache2[layer_idx] = torch.cat([self.value_cache2[layer_idx], value_states], dim=-2)				
			else:
				self.key_cache2.append(key_states)
				self.value_cache2.append(value_states)
		
		out = super().update(key_states, value_states, layer_idx, cache_kwargs) #tuple of (self.key_cache[layer_idx], self.value_cache[layer_idx])		
		if tensors is None: self.save_to_disk(out, layer_idx) #save only first time cause it's slow to save
		self.layers[layer_idx].keys, self.layers[layer_idx].values = torch.empty(0), torch.empty(0)
		return out



class KVCache_legacy(DynamicCache):
	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
		cache_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		tensors = self.load_from_disk(layer_idx)
		if tensors is not None:
			self.key_cache[layer_idx], self.value_cache[layer_idx] = tensors
			if layer_idx < len(self.key_cache2):
				self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], self.key_cache2[layer_idx]], dim=-2)
				self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], self.value_cache2[layer_idx]], dim=-2)
				self.key_cache2[layer_idx] = torch.cat([self.key_cache2[layer_idx], key_states], dim=-2)
				self.value_cache2[layer_idx] = torch.cat([self.value_cache2[layer_idx], value_states], dim=-2)				
			else:
				self.key_cache2.append(key_states)
				self.value_cache2.append(value_states)
		
		out = super().update(key_states, value_states, layer_idx, cache_kwargs) #tuple of (self.key_cache[layer_idx], self.value_cache[layer_idx])		
		if tensors is None: self.save_to_disk(out, layer_idx) #save only first time cause it's slow to save
		self.key_cache[layer_idx], self.value_cache[layer_idx] = torch.empty(0), torch.empty(0)
		return out
