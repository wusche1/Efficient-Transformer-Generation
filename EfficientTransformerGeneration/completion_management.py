
from typing import List, Tuple, Callable, Optional, Dict, Any
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import json
from transformers import PreTrainedModel, PreTrainedTokenizer

def load_gpu_dataset():
    # Get the directory of the current script
    current_dir = Path(__file__).parent
    
    # Construct the path to the JSON file
    json_path = current_dir / "gpu_memory_dataset.json"
    
    # Check if the file exists
    if not json_path.is_file():
        raise FileNotFoundError(f"GPU memory dataset not found at {json_path}")
    
    # Load and return the JSON data
    with open(json_path, "r") as file:
        return json.load(file)

from .gpu_estimations import (
    generate_gpu_usage_estimator,
    measure_memory_usage,
    generate_input_pairs_and_memory_values,
    generate_gpu_usage_estimator_from_input_pairs_and_memory_values,
)

default_chat_template = """
{% for message in messages %}
{% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
{% endfor %}
""".strip()

class CompletionDataset:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        data: pd.DataFrame,
        completion_name: Optional[str] = None,
        gen_length: int = 50,
        system_prompt: str = "You are a helpful assistant.",
        gpu_batch_size: int = 64,
        verbose: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gpu_batch_size = gpu_batch_size
        if tokenizer.chat_template is None:
            tokenizer.chat_template = default_chat_template
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.completion_name = completion_name or f"{model.config._name_or_path}_completions"
        self.gen_length = gen_length
        self.device = model.device
        self.system_prompt = system_prompt
        self.input_pairs: Optional[List[Tuple[int, int]]] = None
        self.memory_values: Optional[List[float]] = None
        self.failed_input_pairs: List[Tuple[int, int]] = []
        self.verbose = verbose
        self.memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2


        model_name = model.config._name_or_path
        gpu_name = torch.cuda.get_device_name()

        gpu_dataset = load_gpu_dataset()
        setting_found = False
        for setting in gpu_dataset:
            if setting["model_name"] == model_name and setting["gpu_name"] == gpu_name and setting["gpu_batch_size"] == gpu_batch_size:
                setting_found = True
                self.gpu_dataset = setting["dataset"]
                break   
        
        if not setting_found:
            raise ValueError("GPU dataset not found for this model and GPU combination")
        

    def __call__(self, generation_kwargs: Dict[str, Any] = {}) -> pd.DataFrame:
        if "input_ids" not in self.data.columns:
            self.tokenize_data()
        
        if not hasattr(self, "beginning_tokens"):
            self.get_template_tokens()
        
        self.complete_all(generation_kwargs=generation_kwargs)
        return self.data
    
    def complete_all(self, generation_kwargs: Dict[str, Any] = {}) -> None:
        sorted_indices = self.data.sort_values("input_ids_length", ascending=False).index
        indices = sorted_indices.tolist()


        with tqdm(total=len(indices)) as pbar:
            input_length = self.data.loc[indices[0], "input_ids_length"]
            current_batch_size = self.get_batchsize(input_length)
            while indices:
                current_indices = indices[:current_batch_size]
                if self.verbose:
                    print(f"Batch size: {current_batch_size}")
                    print(f"Current input length: {input_length}")

                def func_wrapper():
                    return self.complete_indices(current_indices, generation_kwargs=generation_kwargs)
                
                success, memory_used = measure_memory_usage(func_wrapper, return_value=True)
                if success:
                    if self.verbose:
                        print(f"Batch size {current_batch_size} OK")
                        print(f"Memory utilization: {100*memory_used/self.memory_mb:.2f}%")
                    pbar.update(current_batch_size)

                    indices = indices[current_batch_size:]
                    if indices:
                        input_length = self.data.loc[indices[0], "input_ids_length"]
                        current_batch_size = self.get_batchsize(input_length)
                else:
                    if self.verbose:
                        print(f"Batch size {current_batch_size} too large")
                    torch.cuda.empty_cache()
                    current_batch_size = current_batch_size - self.gpu_batch_size

    def get_batchsize(self, input_length: int) -> int:
        find_next_multiple = lambda n, m: n if n % m == 0 else n + (m - n % m)
        next_input_length = find_next_multiple(input_length, self.gen_length)
        next_input_length = str(int(next_input_length))
        if next_input_length in self.gpu_dataset.keys():
            return int(self.gpu_dataset[next_input_length])-self.gpu_batch_size
        else:
            raise ValueError(f"Batch size not found for input length {next_input_length}")

    def complete_indices(self, indices: List[int], generation_kwargs: Dict[str, Any] = {}) -> bool:
        subset = self.data.iloc[indices]
        tokens = subset["input_ids"].tolist()
        
        max_length = max(map(len, tokens))
        padding_result = self.tokenizer.pad(
            {"input_ids": tokens},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        try:
            padded_tokens = padding_result["input_ids"].to(self.device)
            attention_mask = padding_result["attention_mask"].to(self.device)

            beginning_tokens = self.beginning_tokens.unsqueeze(0).expand(padded_tokens.shape[0], -1)
            ending_tokens = self.ending_tokens.unsqueeze(0).expand(padded_tokens.shape[0], -1)
            padded_tokens = torch.cat([beginning_tokens, padded_tokens, ending_tokens], dim=1)
            
            beginning_mask = torch.ones(beginning_tokens.shape, dtype=torch.long, device=self.device)
            ending_mask = torch.ones(ending_tokens.shape, dtype=torch.long, device=self.device)
            attention_mask = torch.cat([beginning_mask, attention_mask, ending_mask], dim=1)

            pad_length = padded_tokens.shape[1]

            with torch.no_grad():
                completions = self.model.generate(
                    input_ids=padded_tokens,
                    attention_mask=attention_mask,
                    max_new_tokens=self.gen_length,
                    **generation_kwargs,
                )

            completions = completions[:, pad_length:]
            decoded = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
            
            self.data.loc[indices, self.completion_name] = decoded
            return True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return False
            raise e

    def get_template_tokens(self) -> None:
        string_1, string_2 = "A", "B"
        convs = [[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": string}
        ] for string in [string_1, string_2]]
        
        conv_tokens = [self.tokenizer.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True) for conv in convs]
        _, i_diff = torch.where(conv_tokens[0] != conv_tokens[1])
        i_diff = i_diff.item()
        
        self.beginning_tokens = conv_tokens[0][0, :i_diff].to(self.device)
        self.ending_tokens = conv_tokens[0][0, i_diff+1:].to(self.device)

    def tokenize_data(self, tokenizer_name: str = "input_ids") -> None:
        prompts = self.data["prompt"].tolist()
        encoded = self.tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        self.data[tokenizer_name] = encoded["input_ids"]
        self.data[f"{tokenizer_name}_length"] = list(map(len, encoded["input_ids"]))

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Load model and tokenizer
    model_name = "Qwen/Qwen-1_8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token="<|endoftext|>")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
#%%
if __name__ == "__main__":
    test_dataset = [{
        "prompt": f"What is 6 plus {i*i}?",
    } for i in range(1000)]

    data_df = pd.DataFrame(test_dataset)

    completion_dataset = CompletionDataset(model, tokenizer, data_df)


    completion_dataset.tokenize_data()
    completion_dataset.get_template_tokens()

    completion_dataset.complete_all(verbose=True)

    completion_dataset.data


#
# %%
