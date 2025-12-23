
import sys
import os

# Ensure the src directory is in the python path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from ollm import Inference, file_get_contents, TextStreamer
import torch

# 1. Initialize Inference Engine
# llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
o = Inference("qwen3-next-80B", device="cuda:0", logging=True)

# 2. Download/Initialize Model Weights
o.ini_model(models_dir="./models/", force_download=False)

# 3. Optimization: Offload Layers
# (optional) offload some layers to CPU for speed boost or memory saving
o.offload_layers_to_cpu(layers_num=38)

# 4. Initialize Cache
# set None if context is small
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

# 5. Initialize Streamer
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

# 6. Prepare Input Messages
sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# 7. Tokenization
# Note: 'reasoning_effort' argument was removed as it is not standard for apply_chat_template
input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# 8. Create Attention Mask
# Explicitly create attention mask to avoid generation warnings/errors
attention_mask = torch.ones_like(input_ids)

# 9. Generation
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# 10. Decode and Print Answer
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
