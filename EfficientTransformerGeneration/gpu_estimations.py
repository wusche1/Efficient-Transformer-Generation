#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from math import isclose
from sklearn.linear_model import LinearRegression
import numpy as np
import gc

def measure_memory_usage(func, return_value = None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    v = func()
    memory = torch.cuda.max_memory_allocated()/ (1024 * 1024)
    if return_value:
        return v, memory
    return memory


def execute_and_measure_memory(func, *args, **kwargs):
    """
    Execute a function and measure its GPU memory utilization.
    
    Args:
    func (callable): The function to execute.
    *args: Positional arguments to pass to the function.
    **kwargs: Keyword arguments to pass to the function.
    
    Returns:
    tuple: A tuple containing (function_return_value, memory_utilization_rate)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")

    # Clear cache and collect garbage to get a clean start
    torch.cuda.empty_cache()
    gc.collect()

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    # Record initial used memory
    initial_used_memory = torch.cuda.memory_allocated()

    # Execute the function and capture its return value
    try:
        return_value = func(*args, **kwargs)
    finally:
        # Measure peak memory usage
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Calculate memory utilization rate
        memory_utilization_rate = peak_memory / total_memory

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

    return return_value, memory_utilization_rate



def generate_text(model, tokenizer, input_length, gen_length, batch_size = 1):
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, input_length), device=model.device)
    with torch.no_grad():
        text = model.generate(input_ids, max_new_tokens = gen_length, min_new_tokens = gen_length, do_sample=True)
        #print(text.shape)

def generate_gpu_usage_estimator(model, tokenizer, gen_tokens, base_input=50, base_batch=64, step_input=20, step_batch=64):
    input_pairs, memory_values = generate_input_pairs_and_memory_values(model, tokenizer, gen_tokens, base_input, base_batch, step_input, step_batch)
    return generate_gpu_usage_estimator_from_input_pairs_and_memory_values(input_pairs, memory_values)

def generate_input_pairs_and_memory_values(model, tokenizer, gen_tokens, base_input=50, base_batch=64, step_input=20, step_batch=64):
    def memory_function(input_length, batch_size):
        return measure_memory_usage(lambda: generate_text(model, tokenizer, input_length, gen_tokens, batch_size))

    input_pairs = [(base_input, base_batch), (base_input + step_input, base_batch), 
                   (base_input, base_batch + step_batch), (base_input + step_input, base_batch + step_batch)]
    memory_values = [memory_function(*pair) for pair in input_pairs]
    return input_pairs, memory_values

def generate_gpu_usage_estimator_from_input_pairs_and_memory_values(input_pairs, memory_values, verbose=False):
    # Prepare data for linear regression
    X = np.array([[1, pair[0], pair[1], pair[0] * pair[1]] for pair in input_pairs])
    y = np.array(memory_values)

    # Solve for coefficients using numpy's least squares solver
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Extract coefficients
    d, a, b, c = coeffs

    # Create the estimator function
    def estimate_memory(input_length, batch_size):
        return max(0, d + a * input_length + b * batch_size + c * input_length * batch_size)

    def max_batch_size(input_length, memory, gpu_batch_size=64, safety_factor=0.8):
        estimated_batch_size = (memory - d - a * input_length) / (b + c * input_length)
        #multiply with safety factor
        estimated_batch_size *= safety_factor
        #round to the nearest multiple of gpu_batch_size
        estimated_batch_size = round(estimated_batch_size / gpu_batch_size) * gpu_batch_size
        return estimated_batch_size

    # Print results
    if verbose:
        print(f"Memory estimation function:")
        print(f"M(i, b) = {d:.4f} + {a:.4f}*i + {b:.4f}*b + {c:.4f}*i*b")

    return estimate_memory, max_batch_size




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

