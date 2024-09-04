#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def measure_memory_usage(func):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    func()
    return torch.cuda.max_memory_allocated()/ (1024 * 1024)

def generate_text(model, tokenizer, input_length, gen_length):
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, input_length), device=model.device)
    total_length = input_length + gen_length
    with torch.no_grad():
        text = model.generate(input_ids, max_new_tokens = gen_length, min_new_tokens = gen_length, do_sample=True)
        print(text.shape)

def forward_pass(model, tokenizer, length):
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, length), device=model.device)
    with torch.no_grad():
        model(input_ids=input_ids)

# Load model and tokenizer
model_name = "Qwen/Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token="<|endoftext|>")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

# Measure memory usage
gen_length = 50
input_lengths = list(range(1, 30))
generation_memory = []
forward_memory = []

for length in input_lengths:
    generation_memory.append(measure_memory_usage(lambda: generate_text(model, tokenizer, length, gen_length)))
#    forward_memory.append(measure_memory_usage(lambda: forward_pass(model, tokenizer, length)))


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(input_lengths, generation_memory, label='Text Generation')
#plt.plot(seq_lengths, forward_memory, label='Forward Pass')
plt.xlabel('Sequence Length')
plt.ylabel('GPU Memory Usage (MB)')
plt.title('GPU Memory Usage: Text Generation vs Forward Pass')
plt.legend()
plt.grid(True)
plt.show()
# %%
generation_memory
# %%
max(generation_memory)
# %%


print(estimate_max_memory_usage(model, tokenizer, 51, 1))
# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizer

def gpu_usage_at_forwardpass(model, tokenizer, seq_length, batch_size, keep_kv_cache=False):
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)
    
    # Create random input
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_length), device=device)
    
    # Run model with batch size of 1
    with torch.no_grad():
        _ = model.forward(input_ids=input_ids, kv_cache=keep_kv_cache)
    
    # Measure peak memory usage during the computation
    peak_memory = torch.cuda.max_memory_allocated(device)

def estimate_max_batch_size_generation(model, tokenizer, input_length, gen_length):
    # Clear cache and measure initial GPU memory usage

    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    
    free_memory = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)

    memory_at_foward_pass = gpu_usage_at_forwardpass(model, tokenizer, input_length, 1)

    memory_for_generation_

# Usage
# Load model and tokenizer
model_name = "Qwen/Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token="<|endoftext|>")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
max_seq_length = 50  # or whatever maximum length you're testing

max_batch_size = estimate_max_batch_size(model, tokenizer, max_seq_length)
print(f"Estimated maximum batch size for sequence length {max_seq_length}: {max_batch_size}")

# Verify the estimation
input_ids = torch.randint(0, tokenizer.vocab_size, (max_batch_size, max_seq_length), device=model.device)
try:
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_seq_length)
    print(f"Successfully generated with batch size {max_batch_size}")
except RuntimeError as e:
    print(f"Error with batch size {max_batch_size}: {e}")
    print("You may need to adjust the safety_factor")
# %%
model.config
# %%
model.generate
# %%
