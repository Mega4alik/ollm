
import sys
import os

# Ensure the src directory is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from ollm import Inference, file_get_contents, TextStreamer
import torch

# Model: Falcon3-MoE-2x7B-Insruct
# Speedtest trigger:
# “script -c "python3 ./example_falcon3_moe.py" ~/Documents/inference_pvstyyle-log3b_20251229.txt”
# “iostat -mx -p nvme1n1 1 > ~/Documents/iostat-pvstyle-log3b_20251229.txt &”

o = Inference("falcon-moe", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=10) # Adjust offloading as needed

past_key_values = o.DiskCache(cache_dir="./kv_cache/")
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are helpful AI assistant"
um = "List planets starting from Mercury"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

attention_mask = torch.ones_like(input_ids)

outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer,
    temperature=0.1,
    do_sample=True
).cpu()

answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
