import torch
from transformers import AutoProcessor, AutoTokenizer #DiffusionGemmaForBlockDiffusion,
from ollm import gpt_oss, gds_loader, diffusion_gemma
from ollm.gds_loader import DenseWeightsLoader, SingleDenseWeightsLoader, MoEWeightsLoader, get_optimal_safetensor_reader, os, json, re


class TempMoEWeightsLoader(DenseWeightsLoader): #needs to be moved to gds_loader
    def __init__(self, path: str, device="cuda:0"):
        self.path = path #<model_dir>
        index_path = os.path.join(path, 'model.safetensors.index.json')
        with open(index_path) as f: indexes = json.load(f)
        self.manifest, self.safetensors = {}, {}
        for manifest_name, filename in indexes["weight_map"].items():
            if "layernorm" in manifest_name or "layer_scalar" in manifest_name: continue
            #match1 = re.search(r"(model\.decoder\.layers\.\d+\.mlp\.experts\.\d+\.)", manifest_name) #TEMP decoder
            match2 = re.search(r"(model\.decoder\.layers\.\d+\.)", manifest_name)
            if match2:
                base = match2.group(1)
                if base not in self.manifest: self.manifest[base] = {}
                attr_path = manifest_name.replace(base, "")
                self.manifest[base][attr_path] = filename

        self.device = torch.device(device)
        self.offloaded_map = {}

    def preload_layer_safetensors(self, base):
        #for filename, x in self.safetensors.items(): x.close() #f.__exit__(None, None, None)
        #del self.safetensors
        #self.safetensors = {}
        for base1 in list(self.manifest.keys()):
            if base1.startswith(base):
                for attr_path, filename in self.manifest[base1].items():
                    if filename not in self.safetensors:
                        filepath = os.path.join(self.path, filename)
                        self.safetensors[filename] = get_optimal_safetensor_reader(filepath) #safe_open(filepath, framework="pt")



model_dir = "/home/mega4alik/ssd/models/diffusiongemma-26B-A4B-it" #google/diffusiongemma-26B-A4B-it
#model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto",)

device = "cpu"
diffusion_gemma.loader = TempMoEWeightsLoader(model_dir, device=device)
#llama.stats = self.stats
model = diffusion_gemma.MyDiffusionGemmaForBlockDiffusion.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
model.eval()

processor = AutoProcessor.from_pretrained(model_dir)
# Prompt
message = [
    {"role": "user", "content": "Why is the sky blue?"}
]

# Process input
input_ids = processor.apply_chat_template(
    message,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)

# Parse output
text = processor.decode(output[0], skip_special_tokens=False)
