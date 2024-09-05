#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

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

# Measure memory usage
gen_length = 50
input_lengths = list(range(1, 30))
generation_memory = []
forward_memory = []

for length in input_lengths:
    generation_memory.append(measure_memory_usage(lambda: generate_text(model, tokenizer, length, gen_length)))
#    forward_memory.append(measure_memory_usage(lambda: forward_pass(model, tokenizer, length)))



# %%


def generate_gpu_usage_estimator(model, tokenizer,gen_tokens, base_input = 10, base_batch = 1, step_input = 10, step_batch = 5):
    base_gpu = measure_memory_usage(lambda: generate_text(model, tokenizer, base_input, gen_tokens, base_batch))
    step_up_input_gpu = measure_memory_usage(lambda: generate_text(model, tokenizer, base_input + step_input, gen_tokens, base_batch))
    step_up_batch_gpu = measure_memory_usage(lambda: generate_text(model, tokenizer, base_input, gen_tokens, base_batch + step_batch))
    input_gpu_slope = (step_up_input_gpu - base_gpu) / step_input
    batch_gpu_slope = (step_up_batch_gpu - base_gpu) / step_batch
    y_intercept = base_gpu - input_gpu_slope * base_input - batch_gpu_slope * base_batch
    def estimate_memory_usage(input_length, batch_size = 1):
        return input_gpu_slope * input_length + batch_gpu_slope * batch_size + y_intercept
    
    return estimate_memory_usage
    
estimate_memory_usage = generate_gpu_usage_estimator(model, tokenizer, gen_length)
# Usage example
predicted_memory =[estimate_memory_usage(input_length) for input_length in input_lengths]

print(f"Predicted memory usage: {predicted_memory}")
print(f"Measured memory usage: {generation_memory}")


# %%
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(input_lengths, generation_memory, label='Text Generation')
plt.plot(input_lengths, predicted_memory, label='Predicted Generation')
#plt.plot(seq_lengths, forward_memory, label='Forward Pass')
plt.xlabel('Sequence Length')
plt.ylabel('GPU Memory Usage (MB)')
plt.title('GPU Memory Usage: Text Generation vs Forward Pass')
plt.legend()
plt.grid(True)
plt.show()


# %%
batch_sizes = [1,2,3,4,5,6,7,8,9,10]
batch_memory_usage = [ measure_memory_usage(lambda: generate_text(model, tokenizer, 10, gen_length, batch_size)) for batch_size in batch_sizes]
# %%
predicted_batch_memory =[estimate_memory_usage(10, batch_size) for batch_size in batch_sizes]
plt.plot(batch_sizes, batch_memory_usage)
plt.plot(batch_sizes, predicted_batch_memory)
# %%
