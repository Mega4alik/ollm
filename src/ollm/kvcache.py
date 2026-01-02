import os, time, shutil, json
import torch
from transformers import DynamicCache
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List

kvikio_available = False
try:
	import kvikio
	import cupy as cp
	from torch.utils.dlpack import from_dlpack, to_dlpack
	kvikio_available = True
except ImportError:
	pass

class oCache:	
	def ini_ocache(self, cache_dir, device, stats):
		if not cache_dir: raise Exception("cache_dir can not be empty. If you are trying to not use DiskCache, simply set past_key_values=None. This will use default DynamicCache")
		self.cache_folder = os.path.join(cache_dir, "kv_cache")
		self.key_cache2, self.value_cache2 = [], []
		if os.path.exists(self.cache_folder): shutil.rmtree(self.cache_folder)
		os.makedirs(self.cache_folder)
		self.device = device
		self.stats = stats

	def load_from_disk(self, layer_idx):
		if kvikio_available:
			return self.load_from_disk_kvikio(layer_idx)

		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		if not os.path.exists(path): return None
		t1 = time.perf_counter()
		tensors = torch.load(path, map_location=self.device)
		if self.stats: self.stats.set("kvload", t1)
		return tensors

	def save_to_disk(self, tensors, layer_idx):
		if kvikio_available:
			return self.save_to_disk_kvikio(tensors, layer_idx)

		t1 = time.perf_counter()
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		tensors = (tensors[0].cpu(), tensors[1].cpu())
		torch.save(tensors, path)
		if self.stats: self.stats.set("kvsave", t1)

	# --- Kvikio / GDS Implementation ---
	def _get_kvikio_paths(self, layer_idx):
		return (
			f"{self.cache_folder}/layer_{layer_idx}_k.bin",
			f"{self.cache_folder}/layer_{layer_idx}_v.bin",
			f"{self.cache_folder}/layer_{layer_idx}_meta.json"
		)

	def save_to_disk_kvikio(self, tensors, layer_idx):
		t1 = time.perf_counter()
		k, v = tensors[0], tensors[1]
		k_path, v_path, meta_path = self._get_kvikio_paths(layer_idx)

		# Metadata
		meta = {
			"shape_k": list(k.shape), "dtype_k": str(k.dtype),
			"shape_v": list(v.shape), "dtype_v": str(v.dtype)
		}
		with open(meta_path, "w") as f: json.dump(meta, f)

		# Save K
		with kvikio.CuFile(k_path, "w") as f:
			# Zero-copy write from GPU tensor
			# We need to ensure tensor is contiguous
			if not k.is_contiguous(): k = k.contiguous()

			# Convert to CuPy via DLPack to get buffer interface
			# Handle bfloat16 by viewing as int16 (preserving bits) since CuPy doesn't support bfloat16
			if k.dtype == torch.bfloat16:
				cupy_k = cp.from_dlpack(to_dlpack(k.view(torch.int16)))
			else:
				cupy_k = cp.from_dlpack(to_dlpack(k))
			f.write(cupy_k)

		# Save V
		with kvikio.CuFile(v_path, "w") as f:
			if not v.is_contiguous(): v = v.contiguous()

			if v.dtype == torch.bfloat16:
				cupy_v = cp.from_dlpack(to_dlpack(v.view(torch.int16)))
			else:
				cupy_v = cp.from_dlpack(to_dlpack(v))
			f.write(cupy_v)

		if self.stats: self.stats.set("kvsave_gds", t1)

	def load_from_disk_kvikio(self, layer_idx):
		k_path, v_path, meta_path = self._get_kvikio_paths(layer_idx)
		if not os.path.exists(meta_path): return None

		t1 = time.perf_counter()
		with open(meta_path, "r") as f: meta = json.load(f)

		# Helper to load one tensor
		def load_one(path, shape, dtype_str):
			# Map string dtype to torch/cupy
			# Ideally we assume consistent dtype (bfloat16/float16)
			# But for safety, we assume model current dtype

			# Calculate bytes
			# This is slightly hacky for dtype size, assuming bfloat16/float16 = 2 bytes
			# Use numpy/cupy to determine size if possible, or mapping
			dtype_map = {
				"torch.float16": cp.float16, "torch.bfloat16": cp.int16, # cupy doesn't fully support bf16 IO sometimes, treat as int16
				"torch.float32": cp.float32
			}
			# Fallback for bf16 string if different
			cp_dtype = dtype_map.get(dtype_str, cp.float16)

			# If it's bfloat16, we treat it as int16 for raw IO to avoid casting issues,
			# then reinterpret in Torch.
			is_bf16 = "bfloat16" in dtype_str
			if is_bf16: cp_dtype = cp.int16

			n_elems = 1
			for s in shape: n_elems *= s

			# Allocate GPU
			with cp.cuda.Device(0): # Default device 0, TODO: use self.device index
				buf = cp.empty(n_elems, dtype=cp_dtype)

			# Read
			with kvikio.CuFile(path, "r") as f:
				f.read(buf)

			# To Torch
			t = from_dlpack(buf.toDlpack())
			if is_bf16: t = t.view(torch.bfloat16)

			return t.view(shape)

		k = load_one(k_path, meta["shape_k"], meta["dtype_k"])
		v = load_one(v_path, meta["shape_v"], meta["dtype_v"])

		if self.stats: self.stats.set("kvload_gds", t1)
		return (k, v)


class KVCache(DynamicCache, oCache): #DiskCache
	def __init__(self, cache_dir="./kv_cache", device="cuda:0", stats=None):
		super().__init__()		
		self.ini_ocache(cache_dir, device, stats)

	@property
	def seen_tokens(self):
		return self.get_seq_length()

	def get_max_length(self) -> Optional[int]:
		return None

	def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
		# Since max_length is None, we just return the new sequence length
		return new_seq_length

	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
		cache_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:

		# Debug fallback for AttributeError if DynamicCache implementation differs
		if not hasattr(self, "key_cache") and hasattr(self, "layers"):
			# Environment-specific DynamicCache uses 'layers' list
			return self._update_layers_variant(key_states, value_states, layer_idx, cache_kwargs)

		# Standard DynamicCache implementation
		tensors = self.load_from_disk(layer_idx)
		is_prefill = key_states.shape[-2] > 1

		if tensors is not None:
			self.key_cache[layer_idx], self.value_cache[layer_idx] = tensors

			if is_prefill:
				# If we loaded the prompt from disk, and we are pre-filling again (key_states is the prompt),
				# we should NOT append key_states to the loaded tensors, as they are likely identical.
				# We just use the disk version.

				# Initialize key_cache2 to empty if needed, as we are starting generation from here.
				if layer_idx >= len(self.key_cache2):
					self.key_cache2.append(torch.empty(0, device=key_states.device, dtype=key_states.dtype))
					self.value_cache2.append(torch.empty(0, device=value_states.device, dtype=value_states.dtype))
				else:
					self.key_cache2[layer_idx] = torch.empty(0, device=key_states.device, dtype=key_states.dtype)
					self.value_cache2[layer_idx] = torch.empty(0, device=value_states.device, dtype=value_states.dtype)

				# Skip super().update() to prevent duplication of the prompt tokens.
				# We return the loaded cache directly.
				# We must ensure to return the expected tensors
				return self.key_cache[layer_idx], self.value_cache[layer_idx]

			elif layer_idx < len(self.key_cache2):
				self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], self.key_cache2[layer_idx]], dim=-2)
				self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], self.value_cache2[layer_idx]], dim=-2)

				# Append NEW token only
				self.key_cache2[layer_idx] = torch.cat([self.key_cache2[layer_idx], key_states], dim=-2)
				self.value_cache2[layer_idx] = torch.cat([self.value_cache2[layer_idx], value_states], dim=-2)
			else:
				# We loaded from disk, but have no memory cache yet.
				# Start accumulating new tokens here.
				self.key_cache2.append(key_states)
				self.value_cache2.append(value_states)

		out = super().update(key_states, value_states, layer_idx, cache_kwargs)

		if tensors is None:
			# Pre-fill Phase: Save the computed prompt cache to disk
			self.save_to_disk(out, layer_idx)

			# IMPORTANT: Do NOT append the prompt to key_cache2 if we just saved it to disk.
			# We want key_cache2 to only hold tokens generated AFTER this point.
			# Initialize with EMPTY tensors for consistency so subsequent updates can append.
			if layer_idx >= len(self.key_cache2):
				self.key_cache2.append(torch.empty(0, device=key_states.device, dtype=key_states.dtype))
				self.value_cache2.append(torch.empty(0, device=value_states.device, dtype=value_states.dtype))
			else:
				# If it already exists (unlikely in pre-fill, but safety first), reset it.
				self.key_cache2[layer_idx] = torch.empty(0, device=key_states.device, dtype=key_states.dtype)
				self.value_cache2[layer_idx] = torch.empty(0, device=value_states.device, dtype=value_states.dtype)

		# Clear VRAM
		self.key_cache[layer_idx], self.value_cache[layer_idx] = torch.empty(0), torch.empty(0)
		return out

	def _update_layers_variant(self, key_states, value_states, layer_idx, cache_kwargs):
		# Handles the case where self.layers[i].keys/.values is used instead of self.key_cache
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
		
		out = super().update(key_states, value_states, layer_idx, cache_kwargs)

		if tensors is None:
			self.save_to_disk(out, layer_idx)
			# Prevent duplication: Initialize memory buffer as empty, since prompt is on disk.
			if layer_idx >= len(self.key_cache2):
				self.key_cache2.append(torch.empty(0, device=key_states.device, dtype=key_states.dtype))
				self.value_cache2.append(torch.empty(0, device=value_states.device, dtype=value_states.dtype))
			else:
				self.key_cache2[layer_idx] = torch.empty(0, device=key_states.device, dtype=key_states.dtype)
				self.value_cache2[layer_idx] = torch.empty(0, device=value_states.device, dtype=value_states.dtype)

		# Clear VRAM
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
		# Legacy implementation, probably unused but kept for ref.
		# Lacks the fix for duplication.
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
		
		out = super().update(key_states, value_states, layer_idx, cache_kwargs)
		if tensors is None: self.save_to_disk(out, layer_idx)
		self.key_cache[layer_idx], self.value_cache[layer_idx] = torch.empty(0), torch.empty(0)
		return out
