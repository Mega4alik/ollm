from ollm import Inference, file_get_contents, TextStreamer
import torch

# Initialize Inference with logging enabled
# Note: "offloading layers to CPU/GPU..." logs appear during initialization/offloading,
# which happens BEFORE the prompt is processed. This is expected behavior as model weights
# are arranged prior to inference.
o = Inference("qwen3-next-80B", device="cuda:0", logging=True)

# Initialize model (downloads if needed)
o.ini_model(models_dir="./models/", force_download=False)

# Optional: Offload layers to CPU for speed boost (or memory savings)
# This moves specific model layers to CPU RAM, readying them for the "Flowing" weights strategy.
o.offload_layers_to_cpu(layers_num=48)

# Set up DiskCache for large context handling
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

# Initialize streamer
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

# Define messages
sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Apply chat template and tokenize
# Note: reasoning_effort argument removed as it's not supported
input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# Generate attention mask
# Required because batch_size=1 and pad_token_id might equal eos_token_id,
# preventing the model from inferring the mask automatically.
attention_mask = torch.ones_like(input_ids)

# Generate response
# Explicitly passing attention_mask to avoid warnings and potential issues
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# Decode and print answer
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
