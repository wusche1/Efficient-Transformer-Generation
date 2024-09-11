
from typing import List, Tuple, Callable, Optional, Dict, Any
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

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
        completion_length: int = 50,
        memory_mb: Optional[float] = None,
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
        self.completion_length = completion_length
        self.device = model.device
        self.memory_mb = memory_mb or torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        self.system_prompt = system_prompt
        self.input_pairs: Optional[List[Tuple[int, int]]] = None
        self.memory_values: Optional[List[float]] = None
        self.failed_input_pairs: List[Tuple[int, int]] = []
        self.verbose = verbose

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

        self.input_pairs, self.memory_values = generate_input_pairs_and_memory_values(
            self.model, self.tokenizer, self.completion_length,
            base_input=50, base_batch=self.gpu_batch_size,
            step_input=20, step_batch=self.gpu_batch_size
        )

        _, get_batchsize = generate_gpu_usage_estimator_from_input_pairs_and_memory_values(
            self.input_pairs, self.memory_values, verbose=self.verbose
        )

        with tqdm(total=len(indices)) as pbar:
            current_batch_size = get_batchsize(self.data.loc[indices[0], "input_ids_length"], self.memory_mb, gpu_batch_size=self.gpu_batch_size)
            while indices:
                input_length = self.data.loc[indices[0], "input_ids_length"]
                input_length, current_batch_size = self.check_input_pair_against_constraints(input_length, current_batch_size)
                current_indices = indices[:current_batch_size]

                def func_wrapper():
                    return self.complete_indices(current_indices, generation_kwargs=generation_kwargs)
                
                success, memory_used = measure_memory_usage(func_wrapper, return_value=True)
                input_pair = (input_length, current_batch_size)

                if success:
                    if self.verbose:
                        print(f"Batch size {current_batch_size} OK")
                        print(f"Memory utilization: {100*memory_used/self.memory_mb:.2f}%")
                    pbar.update(current_batch_size)

                    if input_pair not in self.input_pairs:
                        self.input_pairs.append(input_pair)
                        self.memory_values.append(memory_used)

                    indices = indices[current_batch_size:]
                    if indices:
                        current_input_indices = self.data.loc[indices[0], "input_ids_length"]
                        current_batch_size = get_batchsize(current_input_indices, self.memory_mb, gpu_batch_size=self.gpu_batch_size)
                else:
                    if self.verbose:
                        print(f"Batch size {current_batch_size} too large")
                    self.failed_input_pairs.append(input_pair)
                    torch.cuda.empty_cache()

    def check_input_pair_against_constraints(self, input_length: int, batch_size: int) -> Tuple[int, int]:
        if self.verbose:
            print(f"Input length: {input_length}")
            print(f"Batch size: {batch_size}")
        
        for succeeded_length, succeeded_batch_size in self.input_pairs:
            if succeeded_length >= input_length and succeeded_batch_size >= batch_size:
                memory_used = self.memory_values[self.input_pairs.index((succeeded_length, succeeded_batch_size))]
                mem_utilization = memory_used / self.memory_mb
                batch_size = succeeded_batch_size + self.gpu_batch_size if mem_utilization < 0.8 else succeeded_batch_size
        
        for failed_length, failed_batch_size in self.failed_input_pairs:
            if input_length >= failed_length and batch_size >= failed_batch_size:
                batch_size = failed_batch_size - self.gpu_batch_size
        
        if self.verbose:
            print(f"New batch size: {batch_size}")
        return int(input_length), int(batch_size)

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
                    max_new_tokens=self.completion_length,
                    do_sample=True,
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
