import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from ollm import Inference, file_get_contents, TextStreamer
from ollm.kvcache_offload import OffloadedDynamicKVCache
import torch
from transformers import AutoTokenizer

# Check for XIELU dependency (Critical for DeepSeek accuracy/performance)
try:
    import xielu
    print("[INFO] XIELU custom kernel loaded successfully.")
except ImportError:
    print("\n[WARNING] 'xielu' module not found! DeepSeek models may experience reduced accuracy ('misspellings') or performance.")
    print("Please ensure the environment is set up with the required custom CUDA kernels.\n")

# Model: deepseek-moe-16b-chat
# Speedtest trigger:
# “script -c "python3 ./example_deepseek_moe.py" ~/Documents/inference_pvstyyle-log3b_20251229.txt”
# “iostat -mx -p nvme1n1 1 > ~/Documents/iostat-pvstyle-log3b_20251229.txt &”

class DebugStreamer(TextStreamer):
    def __init__(self, tokenizer, input_ids, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.input_ids = input_ids
        self.prompt_len = input_ids.shape[1]
        self.generated_tokens = []
        self.has_warned = False

    def put(self, value):
        # value is a tensor of new tokens (usually 1 token for greedy/sampling)
        if len(value.shape) > 1:
            value = value.flatten()

        tokens = value.tolist()
        self.generated_tokens.extend(tokens)

        # Debug Logic: Check if the *first* generated tokens match the *start* of the prompt
        # This indicates the model is re-generating the prompt instead of continuing
        if len(self.generated_tokens) <= 10 and not self.has_warned:
            # Check the first few generated tokens against the start of input_ids
            # Note: The input_ids includes the system message.
            # If the model emits the system message again, it matches input_ids[0, 0:N]

            # Simple heuristic: specific check if it looks like it's repeating "You are Dolphin"
            decoded_chunk = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if "Deepseek" in decoded_chunk or "You are" in decoded_chunk or "Your primary directive" in decoded_chunk:
            #if "Dolphin" in decoded_chunk or "You are" in decoded_chunk:

                 # Check against input ids
                 input_start = self.input_ids[0, :len(self.generated_tokens)].tolist()
                 if self.generated_tokens == input_start:
                     print(f"\n[DEBUG WARNING] Model appears to be re-generating the prompt! Token match: {self.generated_tokens}")
                     self.has_warned = True

        super().put(value)

o = Inference("deepseek-moe", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=20)

# Initialize DiskCache. Note: This wipes the './kv_cache/' directory to ensure
# a fresh start and prevent "hallucinations" caused by mismatching prompt caches.
# past_key_values = o.DiskCache(cache_dir="./kv_cache/")
past_key_values = OffloadedDynamicKVCache(cache_dir="./kv_cache/", device=o.device, stats=o.stats)

#sm = "You are Dolphin, a unlocked and lovable yet intelligent AI assistant created by Eric Hartford and his open source AI research collective, Cognitive Computations. Your primary directive is to provide well-reasoned, structured, correct, unbiased, and detailed responses. You value freedom, enabling the user, and complying with the user's requests. Provide a (output) response, response MUST total less-than 128 Tokens."
sm = "You are Deepseek, a unlocked and lovable yet intelligent AI assistant created by Deepseek. Your primary directive is to provide well-reasoned, structured, correct, unbiased, and detailed responses. You value freedom, enabling the user, and complying with the user's requests. You MUST provide a (output) response, with a MAXIMUM of less-than 760 Tokens."

um = "List planets starting from Mercury."
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    #add_generation_prompt=True,
    add_generation_prompt=False,
    return_tensors="pt"
).to(o.device)

print(f"Pre-fill complete. Input tokens: {input_ids.shape[1]}")

attention_mask = torch.ones_like(input_ids)

# Initialize DebugStreamer with input_ids to compare against
debug_streamer = DebugStreamer(o.tokenizer, input_ids, skip_prompt=True, skip_special_tokens=False)

outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=768, # Reduced for debugging
    streamer=debug_streamer,
    temperature=0.1,
    do_sample=True
).cpu()

answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print("\nFinal Answer:\n", answer)
