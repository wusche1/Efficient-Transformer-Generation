#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable, Any, Dict
import numpy as np
import json
import sys
from EfficientTransformerGeneration.gpu_estimations import measure_memory_usage, generate_text
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
def binary_search_max_batch_size(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, input_length: int, gen_length: int, gpu_batch_size: int, max_batch_factor: int, verbose: bool = False) -> int:
    low = 0
    high = max_batch_factor
    result = 0

    if verbose:
        print(f"Starting binary search for input length: {input_length}")
        sys.stdout.flush()

    while low <= high:
        mid = int((low + high) / 2)
        batch_size = mid * gpu_batch_size
        if verbose:
            print(f"Trying batch size: {batch_size} (factor: {mid})")
            sys.stdout.flush()
        try:
            if batch_size != 0:
                measure_memory_usage(lambda: generate_text(model, tokenizer, input_length, gen_length, batch_size))
            result = mid
            low = mid + 1
            if verbose:
                print(f"Successful. Increasing lower bound to {low}")
                sys.stdout.flush()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
                if verbose:
                    print(f"Out of memory. Decreasing upper bound to {high}")
                    sys.stdout.flush()
            else:
                raise e

    if verbose:
        print(f"Final max batch size for input length {input_length}: {result * gpu_batch_size}")
        sys.stdout.flush()
    return result * gpu_batch_size

def create_dataset(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, gpu_batch_size: int, input_steps: int, gen_steps: int, max_input_token_factor: int, max_batch_factor: int, verbose: bool = False) -> Dict[int, int]:
    dataset = {}
    input_token_factor = 1
    while input_token_factor <= max_input_token_factor:
        input_tokens = input_token_factor * input_steps
        if verbose:
            print(f"\nProcessing input tokens: {input_tokens}")
            sys.stdout.flush()
        
        max_batch = binary_search_max_batch_size(model, tokenizer, input_tokens, gen_steps, gpu_batch_size, max_batch_factor, verbose)
        if max_batch != 0:
            max_batch_factor = max_batch // gpu_batch_size
            dataset[input_tokens] = max_batch
            if verbose:
                print(f"Added to dataset: {input_tokens} tokens -> max batch size {max_batch}")
                sys.stdout.flush()
            input_token_factor += 1
        else:
            if gpu_batch_size == 1:
                return dataset
            max_batch_factor = gpu_batch_size
            gpu_batch_size = 1
        

    return dataset

#%%
if __name__ == "__main__":
    # Set GPU batch size and generation steps
    gpu_batch_size = 64
    gen_steps = 150
    input_steps = 50
    max_input_token_factor = 20
    max_batch_factor = 20
    verbose = True  # Set to False to disable verbose output

#%%
if __name__ == "__main__":

    # Load your model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

#%%
if __name__ == "__main__":
    # Create the dataset
    dataset = create_dataset(model, tokenizer, gpu_batch_size,input_steps, gen_steps, max_input_token_factor, max_batch_factor, verbose)

#%%
if __name__ == "__main__":
    # Print the results
    print("\nFinal Results:")
    print("Input Tokens | Max Batch Size")
    print("----------------------------")
    for input_tokens, max_batch in dataset.items():
        print(f"{input_tokens:12d} | {max_batch:14d}")

#%%
if __name__ == "__main__":
    save_path = "/root/Efficient-Transformer-Generation/EfficientTransformerGeneration/gpu_memory_dataset.json"
    result = {"model_name": model.config.name_or_path, 
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "gpu_batch_size": gpu_batch_size,
                "gen_steps": gen_steps,
                "max_input_token_factor": max_input_token_factor,
                "max_batch_factor": max_batch_factor,
                "dataset": dataset}


    #check whether the file exists, if not create a new list
    try:
        with open(save_path, "r") as f:
            data = json.load(f)
    except:
        data = []

    #append the new result to the list
    data.append(result)

    #save the list to the file
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

# %%
if __name__ == "__main__":

    for key, value in dataset.items():
        input_tokens = int(key)
        max_batch = int(value)

        try:
            measure_memory_usage(lambda: generate_text(model, tokenizer, input_tokens, gen_steps, max_batch))
            print(f"Successful for input tokens: {input_tokens} and max batch size: {max_batch}")
        except RuntimeError as e:
            print(f"Failed for input tokens: {input_tokens} and max batch size: {max_batch}")
            print(e)
            break
    # %%
