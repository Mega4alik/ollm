from __future__ import annotations

import os
import zipfile
from typing import Dict, Optional

import requests
import torch
from transformers import AutoProcessor, AutoTokenizer

from ..kvcache import KVCache
from .base import PipelineAdapter
from .registry import register_adapter


_LLAMA_URLS: Dict[str, str] = {
    "llama3-1B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-1B-chat.zip",
    "llama3-3B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-3B-chat.zip",
    "llama3-8B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-8B-chat.zip",
    "gpt-oss-20B": "https://ollm.s3.us-east-1.amazonaws.com/models/gpt-oss-20B.zip",
}

_HF_MODELS: Dict[str, str] = {
    "qwen3-next-80B": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "gemma3-12B": "google/gemma-3-12b-it",
}


@register_adapter(
    [
        "llama3-1B-chat",
        "llama3-3B-chat",
        "llama3-8B-chat",
        "gpt-oss-20B",
        "qwen3-next-80B",
        "gemma3-12B",
    ]
)
class LLMPipelineAdapter(PipelineAdapter):
    """Adapter that encapsulates the original oLLM LLM loading behaviour."""

    def prepare(self, models_dir: str, force_download: bool = False) -> str:
        os.makedirs(models_dir, exist_ok=True)
        model_dir = os.path.join(models_dir, self.model_id)
        if not os.path.exists(model_dir) or force_download:
            if self.model_id in _HF_MODELS:
                self._hf_download(model_dir)
            else:
                self._download_and_unpack(models_dir)
        return model_dir

    def _download_and_unpack(self, models_dir: str) -> None:
        url = _LLAMA_URLS[self.model_id]
        filename = url.split("/")[-1]
        zip_path = os.path.join(models_dir, filename)
        print(f"Downloading {url} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {zip_path}")

        print(f"Unpacking {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(models_dir)
        print(f"Unpacked to {models_dir}")
        os.remove(zip_path)

    def _hf_download(self, model_dir: str) -> None:
        from huggingface_hub import snapshot_download

        repo = _HF_MODELS[self.model_id]
        print(f"Downloading {repo} ...")
        snapshot_download(repo_id=repo, local_dir=model_dir, local_dir_use_symlinks=False)

    def load(self, model_dir: str) -> None:
        print("loading model from", model_dir)
        if self.model_id == "qwen3-next-80B":
            from .. import qwen3_next
            from ..gds_loader import MoEWeightsLoader2

            qwen3_next.loader = MoEWeightsLoader2(model_dir)
            qwen3_next.stats = self.stats
            self.model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
        elif self.model_id == "gemma3-12B":
            from .. import gemma3
            from ..gds_loader import Gemma3Loader

            gemma3.loader = Gemma3Loader(model_dir)
            gemma3.stats = self.stats
            automodel = (
                gemma3.MyGemma3ForConditionalGeneration
                if self.kwargs.get("multimodality")
                else gemma3.MyGemma3ForCausalLM
            )
            self.model = automodel.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_dir)
        elif self.model_id == "gpt-oss-20B":
            from .. import gpt_oss
            from ..gds_loader import GDSWeights

            gpt_oss.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
            gpt_oss.stats = self.stats
            self.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
        else:
            from .. import llama
            from ..gds_loader import GDSWeights

            llama.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
            llama.stats = self.stats
            self.model = llama.MyLlamaForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="cpu",
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
            self.model.clean_layers_weights()

        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def offload_layers_to_cpu(self, **args):
        if hasattr(self.model, "offload_layers_to_cpu"):
            return self.model.offload_layers_to_cpu(**args)
        raise AttributeError("Model does not support CPU offload")

    def offload_layers_to_gpu_cpu(self, **args):
        if hasattr(self.model, "offload_layers_to_gpu_cpu"):
            return self.model.offload_layers_to_gpu_cpu(**args)
        raise AttributeError("Model does not support GPU/CPU offload")

    def disk_cache(self, **args):
        if self.model_id in {"gpt-oss-20B"}:
            print(f"{self.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
            return None
        elif self.model_id == "qwen3-next-80B":
            from ..qwen3_next import Qwen3NextDiskCache

            return Qwen3NextDiskCache(self.model.config, stats=self.stats, **args)
        else:
            return KVCache(stats=self.stats, **args)

    def generate(self, *args, **kwargs):
        if not hasattr(self.model, "generate"):
            raise AttributeError("Model does not expose a generate method")
        return self.model.generate(*args, **kwargs)

    def metadata(self) -> Optional[Dict[str, str]]:
        return {"type": "llm"}
