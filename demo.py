#%%
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('./EfficientTransformerGeneration')
from EfficientTransformerGeneration import CompletionDataset, generate_gpu_usage_estimator
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
text_1 = "Hello, how are you?"
text_2 = "Given, that the sky is blue, what is the color of the sky?"
text_3 = "What is the meaning of life?"
text_4 = "Please explain the following quote: 'The only thing we have to fear is fear itself.'"
text_5 = "What is the capital of France?"
text_6 = "Translate into english: Dunkel war's, der Mond schien helle, schneebedeckt die grüne Flur. Als ein Wagen blitzesschnelle langsam um die Ecke fuhr. Drinnen saßen stehend Leute schweigend ins Gespräch vertieft. Als ein totgeschoss'ner Hase auf der Sandbank Schlittschuh lief."

dataset = []
for i in range(300):
    dataset.append({"prompt": text_1})
    dataset.append({"prompt": text_2})
    dataset.append({"prompt": text_3})
    dataset.append({"prompt": text_4})
    dataset.append({"prompt": text_5})
    dataset.append({"prompt": text_6})

# %%
model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

#%%
c_d = CompletionDataset(model, tokenizer, dataset)
#results = c_d()
# %%
c_d()


# %%
