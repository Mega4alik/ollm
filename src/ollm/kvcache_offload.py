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

class OffloadedDynamicKVCache(DynamicCache):
	"""
	A specialized KVCache implementation that enforces a strict Load-Update-Save-Offload cycle
	for every generation step. This is designed for low-VRAM environments where the full
	KV cache history cannot fit in memory.
	"""
	def __init__(self, cache_dir="./kv_cache", device="cuda:0", stats=None):
		super().__init__()
		self.cache_folder = os.path.join(cache_dir, "kv_cache")
		if os.path.exists(self.cache_folder): shutil.rmtree(self.cache_folder)
		os.makedirs(self.cache_folder)
		self.device = device
		self.stats = stats

		# Ensure we can handle 'layers' attribute if 'key_cache' is missing
		# (compatibility with custom DynamicCache implementations)
		if not hasattr(self, "key_cache"):
			self.key_cache = []
			self.value_cache = []

		# We must track length manually since we don't keep tensors in memory
		self._current_seq_len = 0

	@property
	def seen_tokens(self):
		return self.get_seq_length()

	def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
		"""Returns the sequence length of the cached states."""
		# Override standard behavior which checks len(self.key_cache[0])
		return self._current_seq_len

	def get_max_length(self) -> Optional[int]:
		return None

	def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
		return new_seq_length

	def _get_paths(self, layer_idx):
		return (
			f"{self.cache_folder}/layer_{layer_idx}_k.bin",
			f"{self.cache_folder}/layer_{layer_idx}_v.bin",
			f"{self.cache_folder}/layer_{layer_idx}_meta.json"
		)

	def _save_to_disk(self, k, v, layer_idx):
		if kvikio_available:
			self._save_to_disk_kvikio(k, v, layer_idx)
		else:
			self._save_to_disk_torch(k, v, layer_idx)

	def _load_from_disk(self, layer_idx, device):
		if kvikio_available:
			return self._load_from_disk_kvikio(layer_idx, device)
		else:
			return self._load_from_disk_torch(layer_idx, device)

	def _save_to_disk_torch(self, k, v, layer_idx):
		t1 = time.perf_counter()
		# We save standard torch files for non-kvikio fallback
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		torch.save((k.cpu(), v.cpu()), path)
		if self.stats: self.stats.set("kvsave", t1)

	def _load_from_disk_torch(self, layer_idx, device):
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		if not os.path.exists(path): return None
		t1 = time.perf_counter()
		tensors = torch.load(path, map_location=device)
		if self.stats: self.stats.set("kvload", t1)
		return tensors

	def _save_to_disk_kvikio(self, k, v, layer_idx):
		t1 = time.perf_counter()
		k_path, v_path, meta_path = self._get_paths(layer_idx)

		# Metadata
		meta = {
			"shape_k": list(k.shape), "dtype_k": str(k.dtype),
			"shape_v": list(v.shape), "dtype_v": str(v.dtype)
		}
		with open(meta_path, "w") as f: json.dump(meta, f)

		# Helper to write
		def write_tensor(tensor, path):
			if not tensor.is_contiguous(): tensor = tensor.contiguous()

			# Handle bfloat16
			if tensor.dtype == torch.bfloat16:
				# View as int16 to preserve bits when converting to DLPack
				cupy_t = cp.from_dlpack(to_dlpack(tensor.view(torch.int16)))
			else:
				cupy_t = cp.from_dlpack(to_dlpack(tensor))

			with kvikio.CuFile(path, "w") as f:
				f.write(cupy_t)

		write_tensor(k, k_path)
		write_tensor(v, v_path)

		if self.stats: self.stats.set("kvsave_gds", t1)

	def _load_from_disk_kvikio(self, layer_idx, device):
		k_path, v_path, meta_path = self._get_paths(layer_idx)
		if not os.path.exists(meta_path): return None

		t1 = time.perf_counter()
		with open(meta_path, "r") as f: meta = json.load(f)

		def load_tensor(path, shape, dtype_str):
			# Determine CuPy dtype
			is_bf16 = "bfloat16" in dtype_str
			cp_dtype = cp.int16 if is_bf16 else cp.float16 # Default/Fallback
			if "float32" in dtype_str: cp_dtype = cp.float32

			n_elems = 1
			for s in shape: n_elems *= s

			# Determine device index from string "cuda:0" or torch.device
			dev_idx = 0
			if isinstance(device, str) and "cuda" in device:
				try:
					dev_idx = int(device.split(":")[-1])
				except: pass
			elif isinstance(device, torch.device) and device.type == "cuda":
				if device.index is not None:
					dev_idx = device.index

			with cp.cuda.Device(dev_idx):
				buf = cp.empty(n_elems, dtype=cp_dtype)

			with kvikio.CuFile(path, "r") as f:
				f.read(buf)

			t = from_dlpack(buf.toDlpack())
			if is_bf16: t = t.view(torch.bfloat16)
			return t.view(shape)

		k = load_tensor(k_path, meta["shape_k"], meta["dtype_k"])
		v = load_tensor(v_path, meta["shape_v"], meta["dtype_v"])

		if self.stats: self.stats.set("kvload_gds", t1)
		return (k, v)

	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
		cache_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:

		# 1. Load existing history from disk
		# Note: We ignore self.key_cache/self.layers contents because they are dummy tensors.
		tensors = self._load_from_disk(layer_idx, key_states.device)

		# 2. Update logic
		if tensors is None:
			# Pre-fill (first time seeing this layer)
			k_new, v_new = key_states, value_states
		else:
			k_past, v_past = tensors

			# Check for pre-fill re-entry (Prompt Processing vs Token Generation)
			is_incoming_prefill = key_states.shape[-2] > 1

			if is_incoming_prefill:
				k_new, v_new = k_past, v_past
			else:
				# Generation: Append new token to past
				k_new = torch.cat([k_past, key_states], dim=-2)
				v_new = torch.cat([v_past, value_states], dim=-2)

		# 3. Save UPDATED state back to disk immediately
		self._save_to_disk(k_new, v_new, layer_idx)

		# 4. Update length tracking (using the first layer as reference for global seq len)
		if layer_idx == 0:
			self._current_seq_len = k_new.shape[-2]

		# 5. Populate internal structures with DUMMY tensors to prevent VRAM growth
		# We must maintain list length to satisfy DynamicCache indexing.
		# We use empty tensors or small placeholders.
		dummy_k = torch.empty(0, device=key_states.device)
		dummy_v = torch.empty(0, device=value_states.device)

		# Handle the "layers" vs "key_cache" difference
		if hasattr(self, "layers") and not hasattr(self, "key_cache"):
			# Custom DynamicCache implementation found in some environments
			if layer_idx >= len(self.layers):
				# Expand layers list if needed (though usually pre-filled)
				# Assuming a simple object structure if this path is hit
				class LayerStore: pass
				new_layer = LayerStore()
				new_layer.keys = dummy_k
				new_layer.values = dummy_v
				self.layers.append(new_layer)
			else:
				self.layers[layer_idx].keys = dummy_k
				self.layers[layer_idx].values = dummy_v
		else:
			# Standard DynamicCache
			if len(self.key_cache) <= layer_idx:
				self.key_cache.append(dummy_k)
				self.value_cache.append(dummy_v)
			else:
				self.key_cache[layer_idx] = dummy_k
				self.value_cache[layer_idx] = dummy_v

		# 6. Return the FULL tensors for Attention computation
		# These will be freed by PyTorch once the forward pass for this layer is done,
		# because they are not referenced by self.key_cache/value_cache.
		return k_new, v_new

	def evict(self, layer_idx):
		"""
		Explicitly clear VRAM for this layer.
		Not strictly needed with the update() overwrite, but good for safety.
		"""
		if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
			self.key_cache[layer_idx] = torch.empty(0)
			self.value_cache[layer_idx] = torch.empty(0)

		if hasattr(self, "layers") and len(self.layers) > layer_idx:
			if hasattr(self.layers[layer_idx], "keys"):
				self.layers[layer_idx].keys = torch.empty(0)
				self.layers[layer_idx].values = torch.empty(0)
