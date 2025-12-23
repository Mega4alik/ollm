from ollm import Inference, file_get_contents, TextStreamer
import torch

o = Inference("qwen3-next-80B", device="cuda:0", logging=True) #llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=38) #(optional) offload some layers to CPU for speed boost
past_key_values = o.DiskCache(cache_dir="./kv_cache/") #set None if context is small
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Removed reasoning_effort="minimal" as it's not a standard argument for most tokenizers/templates
# If the specific model template DOES support it, you can add it back to kwargs, but verify first.
input_ids = o.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)

# Create attention_mask manually since apply_chat_template doesn't return it by default
# and the warning indicates it is missing.
attention_mask = torch.ones_like(input_ids)

outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
