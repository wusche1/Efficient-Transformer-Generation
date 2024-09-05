#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def generate_gpu_usage_estimator(model, tokenizer,gen_tokens, base_input = 10, base_batch = 1, step_input = 10, step_batch = 5):
    base_gpu = measure_memory_usage(lambda: generate_text(model, tokenizer, base_input, gen_tokens, base_batch))
    step_up_input_gpu = measure_memory_usage(lambda: generate_text(model, tokenizer, base_input + step_input, gen_tokens, base_batch))
    step_up_batch_gpu = measure_memory_usage(lambda: generate_text(model, tokenizer, base_input, gen_tokens, base_batch + step_batch))
    step_up_batch_input_gpu = measure_memory_usage(lambda: generate_text(model, tokenizer, base_input + step_input, gen_tokens, base_batch + step_batch))
    batch_slope_base = (step_up_batch_gpu - base_gpu) / step_batch
    batch_slope_at_high_input = (step_up_batch_input_gpu - step_up_input_gpu) / step_batch
    batch_slope_slope = (batch_slope_at_high_input - batch_slope_base) / step_input

    #print(f"Base GPU: {base_gpu}")
    #print(f"Step up input GPU: {step_up_input_gpu}")
    input_slope = (step_up_input_gpu - base_gpu) / step_input
    #print(f"Input slope: {input_slope}")

    y_intercept = base_gpu - input_slope * base_input - batch_slope_base * (base_batch-1)
    def estimate_memory_usage(input_length, batch_size = 1):
        batch_slope = batch_slope_base + (input_length - base_input)*batch_slope_slope
        return input_slope * input_length + batch_slope * (batch_size-1) + y_intercept
    
    #assert, that under the 4 conditions where we know the memory usage, the function is correct
    assert base_gpu == estimate_memory_usage(base_input, base_batch), f"Base GPU: {base_gpu}, Estimated: {estimate_memory_usage(base_input, base_batch)}"
    assert step_up_input_gpu == estimate_memory_usage(base_input + step_input, base_batch), f"Step up input GPU: {step_up_input_gpu}, Estimated: {estimate_memory_usage(base_input + step_input, base_batch)}"
    assert step_up_batch_gpu == estimate_memory_usage(base_input, base_batch + step_batch), f"Step up batch GPU: {step_up_batch_gpu}, Estimated: {estimate_memory_usage(base_input, base_batch + step_batch)}"
    assert step_up_batch_input_gpu == estimate_memory_usage(base_input + step_input, base_batch + step_batch), f"Step up batch input GPU: {step_up_batch_input_gpu}, Estimated: {estimate_memory_usage(base_input + step_input, base_batch + step_batch)}"
    return estimate_memory_usage
#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # Load model and tokenizer
    model_name = "Qwen/Qwen-1_8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token="<|endoftext|>")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
#%%
if __name__ == "__main__":

    # Measure memory usage
    gen_length = 50
    input_lengths = list(range(1, 30, 5))
    batch_sizes = [1, 2, 4, 8, 16]


    input_memory = [ measure_memory_usage(lambda: generate_text(model, tokenizer, length, gen_length)) for length in input_lengths]

    batch_momory = [ measure_memory_usage(lambda: generate_text(model, tokenizer, input_lengths[0], gen_length, batch_size)) for batch_size in batch_sizes]
#%%
if __name__ == "__main__":
    estimate_memory_usage = generate_gpu_usage_estimator(model, tokenizer, gen_length)

    predicted_batch_memory = [estimate_memory_usage(input_lengths[0], batch_size) for batch_size in batch_sizes]
    predicted_input_memory = [estimate_memory_usage(length) for length in input_lengths]

    plt.plot(input_lengths, input_memory, label = "Measured input memory")
    plt.plot(input_lengths, predicted_input_memory, label = "Predicted input memory")
    plt.xlabel("Input length")
    plt.ylabel("Memory usage (MB)")
    plt.legend()
    plt.show()

    plt.plot(batch_sizes, batch_momory, label = "Measured batch memory")
    plt.plot(batch_sizes, predicted_batch_memory, label = "Predicted batch memory")
    plt.xlabel("Batch size")
    plt.ylabel("Memory usage (MB)")
    plt.legend()
    plt.show()

#%%
if __name__ == "__main__":
    memory_list_list = []
    for input_length in input_lengths:
        memory_list = []
        for batch_size in batch_sizes:
            memory = measure_memory_usage(lambda: generate_text(model, tokenizer, input_length, gen_length, batch_size))
            memory_list.append(memory)
        memory_list_list.append(memory_list)
#%%
if __name__ == "__main__":
    colors = plt.cm.viridis(np.linspace(0, 1, len(input_lengths)))
    plt.figure(figsize=(12, 6))
    for i, input_length in enumerate(input_lengths):
        print(batch_sizes, memory_list_list[i])
        plt.plot(batch_sizes, memory_list_list[i], label=f"Input length: {input_length}", c = colors[i])
        #now plot a prediciton in the same colour as dotted line
        plt.plot(batch_sizes, [estimate_memory_usage(input_length, batch_size) for batch_size in batch_sizes], linestyle = "--", c = colors[i])
    plt.xlabel("Batch size")
    plt.ylabel("Memory usage (MB)")
    plt.legend()
    plt.show()
#%%
if __name__ == "__main__":
    slopes = []
    for i in range(len(input_lengths)):
        slopes.append((memory_list_list[i][0]-memory_list_list[i][-1])/(batch_sizes[0]-batch_sizes[-1]))
    
    plt.plot(input_lengths, slopes)
# %%
