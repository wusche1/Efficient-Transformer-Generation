#%%

from typing import Callable, Tuple, List, Any
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def measure_memory_usage(func: Callable[[], Any], return_value: bool = False) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    val = func()
    memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    if return_value:
        return val, memory
    return memory

def generate_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, input_length: int, gen_length: int, batch_size: int = 1) -> None:
    input_ids: Tensor = torch.randint(0, tokenizer.vocab_size, (batch_size, input_length), device=model.device)
    attention_mask: Tensor = torch.ones_like(input_ids)
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=gen_length, min_new_tokens=gen_length, do_sample=True, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)

def generate_input_pairs_and_memory_values(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    gen_tokens: int, 
    base_input: int = 50, 
    base_batch: int = 64, 
    step_input: int = 20, 
    step_batch: int = 64
) -> Tuple[List[Tuple[int, int]], List[float]]:
    input_pairs: List[Tuple[int, int]] = [
        (base_input, base_batch),
        (base_input + step_input, base_batch),
        (base_input, base_batch + step_batch),
        (base_input + step_input, base_batch + step_batch)
    ]
    memory_values: List[float] = [
        measure_memory_usage(lambda: generate_text(model, tokenizer, input_length, gen_tokens, batch_size))
        for input_length, batch_size in input_pairs
    ]
    return input_pairs, memory_values

def generate_gpu_usage_estimator_from_input_pairs_and_memory_values(
    input_pairs: List[Tuple[int, int]], 
    memory_values: List[float],
    verbose: bool = False
    ) -> Tuple[Callable[[int, int], float], Callable[[int, float, int, float], int]]:
    X: np.ndarray = np.array([[1, pair[0], pair[1], pair[0] * pair[1]] for pair in input_pairs])
    y: np.ndarray = np.array(memory_values)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    d, a, b, c = coeffs
    if verbose:
        print(f"Memory = {d:.2f} + {a:.2f} * input_length + {b:.2f} * batch_size + {c:.2f} * input_length * batch_size")

    def estimate_memory(input_length: int, batch_size: int) -> float:
        return max(0, d + a * input_length + b * batch_size + c * input_length * batch_size)

    def max_batch_size(input_length: int, memory: float, gpu_batch_size: int = 64, safety_factor: float = 0.8) -> int:
        estimated_batch_size: float = (memory - d - a * input_length) / (b + c * input_length)
        estimated_batch_size *= safety_factor
        return round(estimated_batch_size / gpu_batch_size) * gpu_batch_size

    return estimate_memory, max_batch_size

def generate_gpu_usage_estimator(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    gen_tokens: int, 
    base_input: int = 50, 
    base_batch: int = 64, 
    step_input: int = 20, 
    step_batch: int = 64
) -> Tuple[Callable[[int, int], float], Callable[[int, float, int, float], int]]:
    input_pairs, memory_values = generate_input_pairs_and_memory_values(model, tokenizer, gen_tokens, base_input, base_batch, step_input, step_batch)
    return generate_gpu_usage_estimator_from_input_pairs_and_memory_values(input_pairs, memory_values)




#%%
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    input_pairs = [(50, 64), (70, 64), (50, 128), (70, 128)]
    memory_values = np.array([200, 300, 400, 500])

    estimate_memory, get_batchsize = generate_gpu_usage_estimator_from_input_pairs_and_memory_values(input_pairs, memory_values, verbose=True)

    max_estimated_batch_size = get_batchsize(100, 1000)
    assert type(max_estimated_batch_size) == int
    assert max_estimated_batch_size%64 == 0



#%%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
#%%
if __name__ == "__main__":

    # Measure memory usage
    gen_length = 50
    estimate_memory_usage, get_batchsize = generate_gpu_usage_estimator(model, tokenizer, gen_length)
    
#%%
if __name__ == "__main__":
    input_lengths = np.arange(50, 120+1, 20)
    batch_sizes = np.arange(64, 256+1, 64)
    memory = np.zeros((len(input_lengths), len(batch_sizes)))
    predicted_memory = np.zeros((len(input_lengths), len(batch_sizes)))
    pbar = tqdm(total=len(input_lengths) * len(batch_sizes))
    for i, input_length in enumerate(input_lengths):
        for j, batch_size in enumerate(batch_sizes):
            pbar.update(1)
            #memory[i, j] = measure_memory_usage(lambda: generate_text(model, tokenizer, input_length, gen_length, batch_size))
            predicted_memory[i, j] = estimate_memory_usage(input_length, batch_size)
            print(f"Input length: {input_length}, Batch size: {batch_size}, Measured memory: {memory[i, j]:.2f}, Predicted memory: {predicted_memory[i, j]:.2f}")
    
# %%
if __name__ == "__main__":
    #load the memory from the csv file
    import pandas as pd
    memory = pd.read_csv("memory.csv", index_col=0).values
# %%
if __name__ == "__main__":

    non_zero_indices = memory > 0
    for i, input_length in enumerate(input_lengths):
        plt.plot(batch_sizes[non_zero_indices[i]], memory[i, non_zero_indices[i]], label=f"Input length: {input_length}")
        plt.plot(batch_sizes[non_zero_indices[i]], predicted_memory[i, non_zero_indices[i]], linestyle="--", color="black")
    
    plt.xlabel("Batch size")
    plt.ylabel("Memory utilization (MB)")
    plt.legend()
# %%
if __name__ == "__main__":
    for i, bs in enumerate(batch_sizes):
        plt.plot(input_lengths, memory[:, i], label=f"Batch size: {bs}")
        plt.plot(input_lengths, predicted_memory[:, i], linestyle="--", color="black")
    plt.xlabel("Input length")
    plt.ylabel("Memory utilization (MB)")
    plt.legend()
# %%
if __name__ == "__main__":
    input_pairs = [(inpt, bs) for inpt in input_lengths for bs in batch_sizes]
    memory_values = memory.ravel()
    estimate_memory, get_batchsize = generate_gpu_usage_estimator_from_input_pairs_and_memory_values(input_pairs, memory_values, verbose=True)
# %%
    extreme_batch_sizes = np.array([6,7]) * 64
    input_len = 100
    exptreme_memory = np.zeros_like(extreme_batch_sizes)
    predicted_extreme_memory = np.zeros_like(extreme_batch_sizes)
    for i, bs in enumerate(extreme_batch_sizes):
        exptreme_memory[i] = measure_memory_usage(lambda: generate_text(model, tokenizer, input_len, gen_length, bs))
        predicted_extreme_memory[i] = estimate_memory_usage(input_len, bs)

