import os, requests, zipfile
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from .utils import Stats, file_get_contents
from .gds_loader import GDSWeights, DenseWeightsLoader, MoEWeightsLoader, SingleDenseWeightsLoader
from .kvcache import KVCache

def get_attn_implementation():
	try:
		import flash_attn
		return "flash_attention_2"
	except ImportError:
		print("Warning: flash_attention_2 is not imported. The context length will be limited")
		return None


class Inference:
	def __init__(self, model_id, device="cuda:0", logging=True, multimodality=False):
		self.model_id = model_id
		self.device = torch.device(device)
		self.multimodality = multimodality
		self.stats = Stats() if logging else None

	def download_and_unpack(self, models_dir: str):
		os.makedirs(models_dir, exist_ok=True)
		urls = {
			"gpt-oss-20B": "https://ollm.s3.us-east-1.amazonaws.com/models/gpt-oss-20B.zip"
		}
		url = urls[self.model_id]
		
		# Extract filename from URL
		filename = url.split("/")[-1]
		zip_path = os.path.join(models_dir, filename)

		# Download the file
		print(f"Downloading {url} ...")
		response = requests.get(url, stream=True)
		response.raise_for_status()
		with open(zip_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=8192):
				f.write(chunk)
		print(f"Downloaded to {zip_path}")

		# Unzip
		print(f"Unpacking {zip_path} ...")
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(models_dir)
		print(f"Unpacked to {models_dir}")

		os.remove(zip_path) # Optional: remove the zip file after extraction

	
	def hf_download(self, model_dir):
		from huggingface_hub import snapshot_download
		urls = {
			"llama3-1B-chat": "unsloth/Llama-3.2-1B-Instruct", #meta-llama/
			"llama3-3B-chat": "unsloth/Llama-3.2-3B-Instruct",
			"llama3-8B-chat": "unsloth/Meta-Llama-3.1-8B-Instruct",
			"gpt-oss-20B": "AnuarSh/gpt-oss-20B",
			"qwen3-next-80B": "Qwen/Qwen3-Next-80B-A3B-Instruct",
			"gemma3-12B": "google/gemma-3-12b-it",
			"voxtral-small-24B": "mistralai/Voxtral-Small-24B-2507",
			"dolphin-24B": "dphn/Dolphin3.0-R1-Mistral-24B",
			"qwen3-8b": "mlabonne/Qwen3-8B-abliterated",
			"qwen2-14b": "huihui-ai/Qwen2.5-14B-Instruct-Abliterated",
			"moonlight-16b": "huihui-ai/Moonlight-16B-A3B-Instruct-abliterated",
			"deepseek-moe": "deepseek-ai/deepseek-moe-16b-chat",
			"neural-chat": "Intel/neural-chat-7b-v3-3",
			"falcon-moe": "ehristoforu/Falcon3-MoE-2x7B-Insruct"
		}
		url = urls[self.model_id]
		print(f"Downloading {url} ...")
		snapshot_download(repo_id=url, local_dir=model_dir, local_dir_use_symlinks=False)

	
	def ini_model(self, models_dir="./models/", force_download=False):
		models_list = ["llama3-1B-chat", "llama3-3B-chat", "llama3-8B-chat", "gpt-oss-20B", "qwen3-next-80B", "gemma3-12B", "voxtral-small-24B", "dolphin-24B", "qwen3-8b", "qwen2-14b", "moonlight-16b", "deepseek-moe", "neural-chat", "falcon-moe"]
		if self.model_id not in models_list:
			raise ValueError("Incorrect model id. It must be one of", models_list)
		
		model_dir = os.path.join(models_dir, self.model_id)
		if os.path.exists(model_dir)==False or force_download==True:
			if self.model_id in ["model-from-S3-zip"]:
				self.download_and_unpack(models_dir)
			else:
				self.hf_download(model_dir)

		self.load_model(model_dir)

	
	def load_model(self, model_dir):
		print("loading model from", model_dir)
		if self.model_id=="qwen3-next-80B":
			from . import qwen3_next
			qwen3_next.loader = MoEWeightsLoader(model_dir, device=self.device)
			qwen3_next.stats = self.stats
			self.model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="gemma3-12B":
			from . import gemma3
			gemma3.loader = DenseWeightsLoader(model_dir, device=self.device)
			gemma3.stats = self.stats
			automodel = gemma3.MyGemma3ForConditionalGeneration if self.multimodality else gemma3.MyGemma3ForCausalLM
			self.model = automodel.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.processor = AutoProcessor.from_pretrained(model_dir)
		elif self.model_id=="voxtral-small-24B":
			from . import voxtral
			voxtral.loader = DenseWeightsLoader(model_dir, device=self.device)
			voxtral.stats = self.stats
			self.model = voxtral.MyVoxtralForConditionalGeneration.from_pretrained(model_dir, dtype="auto", device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.processor = AutoProcessor.from_pretrained(model_dir)
			self.tokenizer = self.processor.tokenizer
		elif self.model_id=="gpt-oss-20B":
			from . import gpt_oss
			gpt_oss.loader = GDSWeights(os.path.join(model_dir, "gds_export"), device=self.device)
			gpt_oss.stats = self.stats
			self.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="dolphin-24B":
			from . import mistral
			mistral.loader = SingleDenseWeightsLoader(model_dir, device=self.device) if os.path.exists(os.path.join(model_dir, "model.safetensors")) else DenseWeightsLoader(model_dir, device=self.device)
			mistral.stats = self.stats
			self.model = mistral.MyMistralForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="falcon-moe":
			from . import mixtral
			mixtral.loader = DenseWeightsLoader(model_dir, device=self.device)
			mixtral.stats = self.stats
			self.model = mixtral.MyMixtralForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="qwen2-14b":
			from . import qwen2
			qwen2.loader = DenseWeightsLoader(model_dir, device=self.device)
			qwen2.stats = self.stats
			self.model = qwen2.MyQwen2ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="qwen3-8b":
			from . import qwen3
			qwen3.loader = DenseWeightsLoader(model_dir, device=self.device)
			qwen3.stats = self.stats
			self.model = qwen3.MyQwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="deepseek-moe":
			from . import deepseek
			deepseek.loader = DenseWeightsLoader(model_dir, device=self.device)
			deepseek.stats = self.stats
			# No trust_remote_code=True needed as we use local code
			self.model = deepseek.MyDeepseekForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="moonlight-16b":
			from . import moonlight
			moonlight.loader = DenseWeightsLoader(model_dir, device=self.device)
			moonlight.stats = self.stats
			self.model = moonlight.MyMoonlightForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="neural-chat":
			from . import mistral
			mistral.loader = DenseWeightsLoader(model_dir, device=self.device)
			mistral.stats = self.stats
			self.model = mistral.MyMistralForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		else:
			from . import llama
			llama.loader = SingleDenseWeightsLoader(model_dir, device=self.device) if self.model_id in ["llama3-1B-chat"] else DenseWeightsLoader(model_dir, device=self.device)
			llama.stats = self.stats
			self.model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)

		self.model.eval()
		self.model.to(self.device)
		if not hasattr(self, "tokenizer"): self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

	
	def offload_layers_to_cpu(self, **args):
		self.model.offload_layers_to_cpu(**args)
	
	def offload_layers_to_gpu_cpu(self, **args):
		self.model.offload_layers_to_gpu_cpu(**args)
	
	def DiskCache(self, cache_dir="./kvcache"):
		if self.model_id in ["gpt-oss-20B"]:
			print(f"{self.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
			return None
		elif self.model_id=="qwen3-next-80B":
			from .qwen3_next import Qwen3NextDiskCache
			return Qwen3NextDiskCache(self.model.config, cache_dir=cache_dir, device=self.device, stats=self.stats)
		else:
			return KVCache(cache_dir=cache_dir, device=self.device, stats=self.stats)



class AutoInference(Inference):
	def __init__(self, model_dir, adapter_dir=None, device="cuda:0", logging=True, multimodality=False):
		from peft import PeftModel, LoraConfig, get_peft_model
		self.device = torch.device(device)
		self.stats = Stats() if logging else None
		config = AutoConfig.from_pretrained(model_dir)
		arc = config.architectures[0]
		sharded = self.is_sharded(model_dir)
		if arc == "LlamaForCausalLM":
			self.model_id = "llama3-8B-chat" if sharded else "llama3-1B-chat"
		elif arc == "Gemma3ForConditionalGeneration":
			self.model_id = "gemma3-12B"
			#multimodality = True
		elif arc == "Gemma3ForCausalLM": self.model_id = "gemma3-12B"
		else:
			raise ValueError("This model type is not supported")
			
		self.multimodality = multimodality
		self.load_model(model_dir) #peft_config.base_model_name_or_path
		if adapter_dir:
			peft_config = LoraConfig.from_pretrained(adapter_dir)
			self.model = get_peft_model(self.model, peft_config)   # this creates LoRA modules with grad enabled
			self.model.load_adapter(adapter_dir, adapter_name="default")
			#self.model = self.model.model #?

	def is_sharded(self, model_dir):
		files = os.listdir(model_dir)
		is_sharded = any("index.json" in f for f in files)
		return is_sharded
