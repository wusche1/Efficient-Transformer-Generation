#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Cache
import matplotlib.pyplot as plt
import numpy as np
from gpu_estimations import execute_and_measure_memory

def measure_memory_usage(func):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    func()
    return torch.cuda.max_memory_allocated()/ (1024 * 1024)

def generate_text(model, tokenizer, input_length, gen_length, batch_size = 1):
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, input_length), device=model.device)
    with torch.no_grad():
        text = model.generate(input_ids, max_new_tokens = gen_length, min_new_tokens = gen_length, do_sample=True)
        print(text.shape)

def forward_pass(model, tokenizer, length):
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, length), device=model.device)
    with torch.no_grad():
        model(input_ids=input_ids)
#%%
# Load model and tokenizer
model_name = "Qwen/Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token="<|endoftext|>")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
#%%


mem_util = []
batch_sizes = []
input_length = 10
gen_tokens = 50
for i in range(1, 500,10):
    batch_size = i
    batch_sizes.append(batch_size)
    mem_util.append(execute_and_measure_memory(lambda: generate_text(model, tokenizer, input_length, gen_tokens, batch_size)))



# %%
plt.plot(batch_sizes[:-1], mem_util)
# %%
mem_util[-1]
# %%
#print out reserved memory
print(torch.cuda.memory_reserved()/torch.cuda.get_device_properties(0).total_memory)
# %%
