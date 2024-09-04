#%%
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
model_name = "Qwen/Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, pad_token="<|endoftext|>"
)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
    device
)
# %%

def estimate_max_batch_size(model, seq_length, dtype=torch.float32):
    # Get GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - reserved_memory - allocated_memory

    # Estimate model memory (this is a simplified estimation)
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Estimate memory per sample (this needs to be adjusted based on your specific model)
    # Use torch.empty to create a tensor of the right dtype and get its element size
    sample_memory = seq_length * torch.empty(1, dtype=dtype).element_size() * model.config.hidden_size

    # Estimate max batch size
    max_batch_size = (free_memory - model_memory) // sample_memory

    return max_batch_size

seq_length = 512  # adjust based on your needs
max_batch = estimate_max_batch_size(model, seq_length)
print(f"Estimated max batch size: {max_batch}")

inut = 
# now, lets test the 
# %%
