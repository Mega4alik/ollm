from ollm import Inference, file_get_contents
from ollm.gds_loader import GDSWeights
from ollm.utils import Stats
from ollm.kvcache import KVCache
from ollm.gpt_oss import MyGptOssForCausalLM
from ollm import gpt_oss, gds_loader, llama

import torch, os
from datetime import datetime
from transformers import AutoTokenizer, TextStreamer, DynamicCache

def inference_chat():
	#sm, um, max_new_tokens = "You are helpful AI assistant", "List planets starting from Mercury", 100
	sm, um, max_new_tokens = "[CHATS]:\n"+file_get_contents("./temp/chats.txt")+"[/END CHATS]", "Analyze chats above and write top 10 most popular questions (translate to English).", 500
	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
	input_ids = tokenizer.apply_chat_template(messages, tokenize=True, reasoning_effort="minimal", add_generation_prompt=True, return_tensors="pt", return_dict=False).to(device)
	text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
	with torch.no_grad():
		past_key_values = None #KVCache(cache_dir="/media/mega4alik/ssd/kv_cache/") #DynamicCache(offloading=True)
		print("\n\nGenerate started.", datetime.now().strftime("%H:%M:%S"), "input_ids.shape:", input_ids.shape)
		outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, past_key_values=past_key_values, use_cache=True, streamer=text_streamer).detach().cpu()
		answer = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
		print(answer)

#=======================================================
if 1==1:
	device = torch.device("cuda:0")
	model_dir = "/media/mega4alik/ssd/models/gpt-oss-20B/"
	print("loading", model_dir)
	gpt_oss.loader = GDSWeights(model_dir+"gds_export/", device="cuda:0")
	stats = Stats()
	gpt_oss.stats, gds_loader.stats = stats, stats
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = MyGptOssForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True) #, attn_implementation="flash_attention_2"
	#model.clean_layers_weights()
	model.eval()
	model.to(device)
	model.offload_layers_to_cpu(layers_num=12)
	inference_chat()