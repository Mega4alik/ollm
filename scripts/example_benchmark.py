
import sys
import os
import time
import torch

# Ensure the src directory is in the python path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from ollm import Inference, TextStreamer

# --- PRE-FLIGHT DIAGNOSTICS ---
print("\n" + "="*40)
print("--- OLLM SYSTEM INITIALIZATION ---")
print("="*40)

# Check for Flash Attention
try:
    import flash_attn
    print(f"✅ Flash Attention package found: {flash_attn.__version__}")
except ImportError:
    print("❌ Flash Attention package NOT found in PYTHONPATH.")

# Wall clock start
start_wall_time = time.time()

# 1. INITIALIZE INFERENCE ENGINE
try:
    # Note: 'use_kvikio' is not currently supported in the Inference class
    # and has been commented out.
    o = Inference(
        model_id="qwen3-next-80B",
        device="cuda:0",
        # use_kvikio=True,
        logging=True
    )
except TypeError as e:
    print(f"⚠️ Initialization Note: {e}")
    print("Attempting fallback initialization...")
    o = Inference(model_id="qwen3-next-80B", device="cuda:0", logging=True)

# 2. INITIALIZE FROM LOCAL MODELS DIR
o.ini_model(
    models_dir="./models/",
    force_download=False
)

# 3. CONFIGURE CPU OFFLOADING
# Offloading 38 layers to CPU as requested (adjusted from 48 to match previous valid example if needed)
o.offload_layers_to_cpu(layers_num=38)

# 4. STORAGE-BACKED KV CACHE
# Using requested relative path
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

# 5. PREPARE STREAMER & INPUTS
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are helpful AI assistant"
um = "List all planets in the solar system starting from Mercury, Then seperatly simply list any newly found planets (Not Dwarfs), Planets must be disputed or under review and MUST be accepted as traditional planets. If information on a Dwarf is listed a very brief 1 sentance explanation can be given. "
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Note: reasoning_effort is not a standard argument for apply_chat_template
input_ids = o.tokenizer.apply_chat_template(
    messages,
    # reasoning_effort="minimal",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# Create attention mask manually to avoid warnings
attention_mask = torch.ones_like(input_ids)

# --- GENERATION PHASE ---
print("\n" + "="*40)
print("--- STARTING INFERENCE ---")
print("="*40)

gen_start = time.time()

outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=96,
    streamer=text_streamer
)

gen_end = time.time()

# --- POST-PROCESSING & METRICS ---
# Ensure answer is decoded and printed
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)

# Time & Speed Calculations
total_script_time = time.time() - start_wall_time
inference_duration = gen_end - gen_start
tokens_generated = len(outputs[0]) - len(input_ids[0])
tokens_per_sec = tokens_generated / inference_duration if inference_duration > 0 else 0

print(f"\n\n" + "="*40)
print("PERFORMANCE SUMMARY")
print("="*40)
print(f"Total Script Runtime:  {total_script_time:.2f}s")
print(f"Inference Duration:    {inference_duration:.2f}s")
print(f"Tokens Generated:      {tokens_generated}")
print(f"Throughput Speed:      {tokens_per_sec:.2f} tokens/sec")

# Handle VRAM metric gracefully if on CPU or invalid device
try:
    if "cuda" in str(o.device):
        device_idx = int(str(o.device).split(":")[-1])
        print(f"VRAM Peak (GPU {device_idx}):     {torch.cuda.max_memory_allocated(device_idx) / 1024**2:.2f} MB")
except Exception:
    print("VRAM Peak:             N/A (CPU or error)")

print("="*40)
