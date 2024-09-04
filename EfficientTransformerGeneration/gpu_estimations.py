#%%
import torch
from typing import Callable
#%%


def estimate_max_batch_size_for_generation(
    model: torch.nn.Module,
    tokenizer: Callable,
    seq_length: int,
    max_new_tokens: int,
    safety_factor: float = 0.85
) -> int:
    device = next(model.parameters()).device
    
    # Clear cache and measure initial GPU memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)
    
    # Create random input
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_length), device=device)
    
    # Run model with batch size of 1
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    
    # Measure peak memory usage during the computation
    peak_memory = torch.cuda.max_memory_allocated(device)
    
    # Calculate memory used for one sample
    memory_per_sample = peak_memory - initial_memory
    
    # Get total available GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = total_memory - initial_memory
    
    # Estimate max batch size
    estimated_max_batch = int((available_memory / memory_per_sample) * safety_factor)
    
    return max(1, estimated_max_batch)  # Ensure we return at least 1
# %%

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seq_length = 512
    max_new_tokens = 512
    max_batch = estimate_max_batch_size_for_generation(model, tokenizer, seq_length, max_new_tokens)
    print(f"Estimated max batch size: {max_batch}")

    input = torch.randint(0, tokenizer.vocab_size, (max_batch+4,seq_length), device=device)
    with torch.no_grad():
        out = model.generate(input_ids=input, max_new_tokens=max_new_tokens)
        del out
        torch.cuda.empty_cache()
    # %%
