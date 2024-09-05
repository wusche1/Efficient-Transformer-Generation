
#%%
import torch
import numpy as np
import pandas as pd

default_chat_template ="""{% for message in messages %}
{% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
{% endfor %}""".strip()

def get_template_tokens(tokenizer, system_prompt):
    if tokenizer.chat_template is None:
        tokenizer.chat_template = default_chat_template
    
    string_1 = "A"
    string_2 = "B"

    convs = [[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": string}] for string in [string_1, string_2]
    ]
    conv_tokens = [tokenizer.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True) for conv in convs]
    _,i_diff =torch.where(conv_tokens[0] != conv_tokens[1])
    i_diff = i_diff.item()
    beginning_tokens = conv_tokens[0][0,:i_diff]
    ending_tokens = conv_tokens[0][0,i_diff+1:]

    return beginning_tokens, ending_tokens


def tokenize_data(data_df, tokenizer, tokenizer_name = "tokenized"):
    prompts = data_df["prompt"].tolist()
    encoded = tokenizer.batch_encode_plus(
        prompts,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding=False,
        truncation=False,W
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
    # Your large list of strings
#%%
if __name__ == "__main__":
    test_dataset = [{
        "prompt": f"I am {i} years old",
    } for i in range(1000)]

    data_df = pd.DataFrame(test_dataset)
    tokenize_data(data_df, tokenizer)

    print(data_df.head())
#%%
if __name__ == "__main__":
    beginning_tokens, endign_tokens = get_template_tokens(tokenizer, "I am 10 years old")
    # %%
beginning_tokens
# %%
endign_tokens

# %%
