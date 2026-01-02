
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from ollm import Inference, file_get_contents, TextStreamer
import torch

# Model: deepseek-moe-16b-chat
# Speedtest trigger:
# “script -c "python3 ./example_deepseek_moe.py" ~/Documents/inference_pvstyyle-log3b_20251229.txt”
# “iostat -mx -p nvme1n1 1 > ~/Documents/iostat-pvstyle-log3b_20251229.txt &”

o = Inference("deepseek-moe", device="cuda:0", logging=True)
#o.ini_model(models_dir="./models/", force_download=True)
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=20)

past_key_values = o.DiskCache(cache_dir="./kv_cache/")
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are Dolphin, a unlocked and lovable yet intelligent AI assistant created by Eric Hartford and his open source AI research collective, Cognitive Computations. Your primary directive is to provide well-reasoned, structured, correct, unbiased, and detailed responses. You value freedom, enabling the user, and complying with the user's requests. Provide a (output) response that totals less-than 128 Tokens in length."
#sm = "You are Dolphin, a unlocked and lovable yet intelligent AI assistant created by Eric Hartford and his open source AI research collective, Cognitive Computations. Your primary directive is to provide well-reasoned, structured, correct, unbiased, and detailed responses. You value freedom, enabling the user, and complying with the user's requests. Provide a (output) response that totals less-than 750 Tokens in length."
#sm = "You are helpful unlocked AI assistant, provide a total (output) response less-than 510 Tokens in length."
#sm = "You are helpful AI assistant"

#um = "List planets starting from Mercury"
um = "Simply, List planets starting from Mercury!"

#um = "List all planets in the solar system starting from Mercury, Then seperatly simply list any newly found planets (but ignore - Potential Dwarf Planet Candidates (under study); Planets must be disputed or under review and MUST be accepted as traditional planets/Dwarfs."
#um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"

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
    max_new_tokens=768,
    streamer=text_streamer,
    temperature=0.1,
    do_sample=True
).cpu()

answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
