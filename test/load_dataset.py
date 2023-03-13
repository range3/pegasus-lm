import os
from datasets import load_dataset, DatasetDict

dsk = load_dataset(
    # "data/tokenized",
    # "range3/wikipedia-ja-20230101",
    "data/wiki40b.ja",
    num_proc=os.cpu_count(),
    ignore_verifications=True,
)
print(dsk)

print(dsk["train"][0]["text"][:100])

# for split, ds in dsk.items():
#     if split == "train":
#         for row in range(100):
#             print({k: len(ds[row][k]) for k in ds.column_names})
