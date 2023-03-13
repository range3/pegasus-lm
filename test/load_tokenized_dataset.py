import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from pegasuslm import PegasusGPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("model/50KV_200MS")

ds = load_dataset(
    "data/tokenized",
    split="test",
    num_proc=os.cpu_count(),
    ignore_verifications=True,
)

for row in range(100):
    tokens = ds[row]["input_ids"] 
    print(tokens)
    print(tokenizer.decode(tokens))
    print("")
