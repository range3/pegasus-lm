from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

# from pprint import pprint as print

dsk = load_dataset("data/narou_small_tokenized")

print(dsk)

tokenizer = AutoTokenizer.from_pretrained("model/50KV_20MS")
collator = DataCollatorWithPadding(tokenizer, True)


@dataclass
class DataCollatorForGPT2:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_attention_mask: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # result = super().__call__(features)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="np",
            return_attention_mask=self.return_attention_mask,
        )
        batch["labels"] = batch["input_ids"]
        return batch


def dummy_collator(features):
    return features


collator = DataCollatorForGPT2(tokenizer)
# collator = default_data_collator
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# print(collator(dsk["train"][0:8]).items())

# print(dsk["train"][:8])

print(
    {
        k: v
        for k, v in collator(
            [{"input_ids": dsk["train"][row]["input_ids"]} for row in range(8)]
        ).items()
    }
)
