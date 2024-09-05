
#%%
import torch
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('.'))
from gpu_estimations import generate_gpu_usage_estimator
#%%

default_chat_template ="""{% for message in messages %}
{% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
{% endfor %}""".strip()

class CompletionDataset:
    def __init__(self, model, tokenizer, data, completion_name = None, completion_length = 50, memory_mb = None, system_prompt = "You are a helpful assistant."):
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer.chat_template is None:
            tokenizer.chat_template = default_chat_template
        self.data = data
        if completion_name is None:
            model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else "model"
            completion_name = f"{model_name}_completions"
        self.completion_name = completion_name
        self.completion_length = completion_length
        self.device = model.device
        if memory_mb is None:
            device = model.device
            free_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2 - torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
        self.memory_mb = free_mb
        self.system_prompt = system_prompt
    
    def complete_all(self):
        sorted_indeces = self.data.sort_values("input_ids_length").index
        indeces = sorted_indeces.tolist()
        _, get_batchsize = generate_gpu_usage_estimator(self.model, self.tokenizer, self.completion_length)
        known_ok_batch_size = np.inf
        current_batch_size = get_batchsize(indeces[0])
        while len(indeces) > 0:
            
            recommended_batch_size = self.get_batchsize(indeces[0])
            indeces = indeces[recommended_batch_size:]
            if not self.complete_indeces(indeces):


    def complete_indeces(self, indeces):
        subset = self.data.iloc[indeces]

        tokens = subset["input_ids"].tolist()
        
        # Pad tokens to the same length to the left
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

            # Add the beginning tokens to the left and the ending tokens to the right
            beginning_tokens = self.beginning_tokens.unsqueeze(0).expand(padded_tokens.shape[0], -1)
            ending_tokens = self.ending_tokens.unsqueeze(0).expand(padded_tokens.shape[0], -1)
            padded_tokens = torch.cat([beginning_tokens, padded_tokens, ending_tokens], dim=1)
            
            # Update attention mask to include beginning and ending tokens
            beginning_mask = torch.ones(beginning_tokens.shape, dtype=torch.long, device=self.device)
            ending_mask = torch.ones(ending_tokens.shape, dtype=torch.long, device=self.device)
            attention_mask = torch.cat([beginning_mask, attention_mask, ending_mask], dim=1)

            batch_size = padded_tokens.shape[0]
            kv_cache_batched = tuple(
                tuple(tensor.expand(batch_size, *tensor.shape[1:]) for tensor in layer)
                for layer in self.beginning_tokens_kv_cache
            )

            pad_length = padded_tokens.shape[1]

            # Generate completions
            with torch.no_grad():
                completions = self.model.generate(
                    input_ids=padded_tokens,
                    attention_mask=attention_mask,
                    max_new_tokens=self.completion_length,
                    do_sample=True,
                    #past_key_values=kv_cache_batched
                )

            # Extract the completions
            completions = completions[:, pad_length:]
            
            # Decode the completions
            decoded = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
            
            # Add the completions to the data
            self.data.loc[indeces, self.completion_name] = decoded
            return True
        except Exception as e:
            #check if error us due to gpu memory
            if "CUDA out of memory" in str(e):
                return False
            else:
                raise e

            return False
    def get_template_tokens(self):
        tokenizer = self.tokenizer
        string_1 = "A"
        string_2 = "B"

        convs = [[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": string}] for string in [string_1, string_2]
        ]
        conv_tokens = [tokenizer.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True) for conv in convs]
        _,i_diff =torch.where(conv_tokens[0] != conv_tokens[1])
        i_diff = i_diff.item()
        
        beginning_tokens = conv_tokens[0][0,:i_diff].to(self.device)
        ending_tokens = conv_tokens[0][0,i_diff+1:].to(self.device)

        self.beginning_tokens = beginning_tokens
        self.ending_tokens = ending_tokens

        self.beginning_tokens_kv_cache = kv_cache = model.forward(beginning_tokens, return_dict=True).past_key_values

    def tokenize_data(self, tokenizer_name = "input_ids"):
        data_df = self.data
        tokenizer = self.tokenizer

        prompts = data_df["prompt"].tolist()
        encoded = tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        encoded_length = list(map(len, encoded["input_ids"]))
        data_df[tokenizer_name] = encoded["input_ids"]
        data_df[tokenizer_name + "_length"] = encoded_length


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
        "prompt": f"What is 6 plus {i}",
    } for i in range(1000)]

    data_df = pd.DataFrame(test_dataset)

    completion_dataset = CompletionDataset(model, tokenizer, data_df)
# %%

completion_dataset.tokenize_data()
# %%
completion_dataset.get_template_tokens()
# %%
completion_idx = list(range(10))
completion_dataset.complete_indeces(completion_idx)
# %%
comp = completion_dataset.data["Qwen/Qwen-1_8B-Chat_completions"][:10]
# %%


for c in comp:
    print(c)

# %%
