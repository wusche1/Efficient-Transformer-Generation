#%%
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('./EfficientTransformerGeneration')
from EfficientTransformerGeneration import CompletionDataset, generate_gpu_usage_estimator
import torch
from tqdm import tqdm
from time import time
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
# %%
batch_size = [64, 128, 256]
times = []
vocab_size = 50256
n_input_tokens = 100
n_new_tokens = 100
for bs in batch_size:
    print(f"Batch size: {bs}")
    input_ids = torch.randint(0, vocab_size, (bs, n_input_tokens)).to(device)
    start = time()
    generated = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=n_new_tokens, min_new_tokens=n_new_tokens, do_sample=True)
    print(generated.shape)
    end = time()
    times.append(end-start)
#%%
import matplotlib.pyplot as plt
plt.plot(batch_size, times)
plt.xlabel("Batch size")
plt.ylabel("Time (s)")
plt.show()
created_tokens = [n_new_tokens * bs for bs in batch_size]
plt.figure()
plt.plot(batch_size, created_tokens)
plt.xlabel("Batch size")
plt.ylabel("Number of tokens created")
plt.show()
tokens_per_second = [n/t for n, t in zip(created_tokens, times)]
plt.figure()
plt.plot(batch_size, tokens_per_second)
plt.xlabel("Batch size")
plt.ylabel("Tokens per second")
plt.show()



# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EfficientTransformerGeneration.gpu_estimations import measure_memory_usage, generate_text

# Define ranges for batch size and input length
batch_sizes = np.arange(64, 320+1, 64)
input_lengths = np.arange(1, 60+1, 5)

# Fixed parameters
gen_length = 30

# Create meshgrid for batch sizes and input lengths
BS, IL = np.meshgrid(batch_sizes, input_lengths)

# Initialize memory array
memory = np.zeros_like(BS, dtype=float)
# %%
# Measure memory usage for each combination
pbar = tqdm(total=len(input_lengths) * len(batch_sizes))
for i, il in enumerate(input_lengths):
    for j, bs in enumerate(batch_sizes):
        pbar.update(1)
        memory[i, j] = measure_memory_usage(lambda: generate_text(model, tokenizer, il, gen_length, bs))

# Prepare data for linear regression
X = np.column_stack((BS.ravel(), IL.ravel()))
y = memory.ravel()

# %%

# Create a meshgrid for the fitted plane


# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the measured data points
scatter = ax.scatter(BS, IL, memory, c=memory, cmap='viridis')

# Plot the fitted plane

# Customize the plot
ax.set_xlabel('Batch Size')
ax.set_ylabel('Input Length')
ax.set_zlabel('Memory Usage')
ax.set_title('Memory Usage vs Batch Size and Input Length')

# Add a color bar
fig.colorbar(scatter, ax=ax, label='Memory Usage')

# Print the coefficients and intercept of the linear regression
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
print(f"R-squared score: {reg.score(X, y)}")

plt.show()
# %%
for i, bs in enumerate(batch_sizes):
    plt.plot(input_lengths, memory[:, i], label=f"Batch size: {bs}")
plt.xlabel("Input length")
plt.ylabel("Memory usage (MB)")
plt.legend()
# %%
for i, il in enumerate(input_lengths):
    plt.plot(batch_sizes, memory[i], label=f"Input length: {il}")
plt.xlabel("Batch size")
plt.ylabel("Memory usage (MB)")
plt.legend()
# %%