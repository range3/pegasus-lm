import os
from typing import Optional, List, Dict
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from huggingface_hub import snapshot_download
from pegasuslm.gpt2.datasets import wiki40b
from transformers import AutoTokenizer, BatchEncoding
from pegasuslm import PegasusGPT2Tokenizer
from multiprocessing import Pool


def load_dataset_from_local_snapshot(dataset_name: str, **load_dataset_kwargs):
    local_dir = snapshot_download(
        dataset_name,
        repo_type="dataset",
        local_files_only=True,
    )
    print(local_dir)
    return load_dataset(local_dir, **load_dataset_kwargs)


def tokenize_texts(
    texts: List[str],
    tokenizer: PegasusGPT2Tokenizer,
    add_special_tokens: bool = True,
    **kwargs,
):
    return tokenizer(
        texts,
        return_token_type_ids=False,
        return_attention_mask=False,
        add_special_tokens=add_special_tokens,
        verbose=False,
    )


def chunk_tokens(tokens: BatchEncoding, chunk_size: int):
    return {
        k: [
            v[i : i + chunk_size]
            for v in b
            for i in range(0, len(v), chunk_size)
            if not (i != 0 and len(v) - i < 10)  # drop the small remainder
        ]
        for k, b in tokens.items()
    }


def preprocess_range3_wiki40b_ja(
    num_proc: int,
    cache_dir: str,
    tokenizer: Optional[PegasusGPT2Tokenizer] = None,
    ignore_verifications: bool = True,
):
    dsd = load_dataset_from_local_snapshot(
        "range3/wiki40b-ja",
        num_proc=num_proc,
        cache_dir=cache_dir,
        ignore_verifications=ignore_verifications,
    )

    process = lambda texts: {"text": [wiki40b.preprocess(text) for text in texts]}

    if tokenizer is not None:
        process = lambda texts: chunk_tokens(
            tokenize_texts([wiki40b.preprocess(text) for text in texts], tokenizer),
            tokenizer.model_max_length,
        )

    dsd = dsd.map(
        process,
        batched=True,
        input_columns="text",
        remove_columns=dsd["train"].column_names,
        num_proc=num_proc,
    )
    return dsd


def preprocess_range3_narou20(
    num_proc: int,
    cache_dir: str,
    tokenizer: Optional[PegasusGPT2Tokenizer] = None,
    ignore_verifications: bool = True,
):
    dsd = load_dataset_from_local_snapshot(
        "range3/narou20",
        num_proc=num_proc,
        cache_dir=cache_dir,
        ignore_verifications=ignore_verifications,
    )
    dsd = dsd.remove_columns([c for c in dsd["train"].column_names if c != "text"])
    if tokenizer is not None:
        dsd = dsd.map(
            lambda texts: chunk_tokens(
                tokenize_texts(texts, tokenizer),
                tokenizer.model_max_length,
            ),
            batched=True,
            input_columns="text",
            remove_columns=["text"],
            num_proc=num_proc,
        )
    return dsd


def preprocess_range3_cc100_ja(
    num_proc: int,
    cache_dir: str,
    tokenizer: Optional[PegasusGPT2Tokenizer] = None,
    ignore_verifications: bool = True,
):
    dsd = load_dataset_from_local_snapshot(
        "range3/cc100-ja",
        num_proc=num_proc,
        cache_dir=cache_dir,
        ignore_verifications=ignore_verifications,
    )
    print(dsd)

    process = lambda texts: {
        "text": list(filter(None, [text.strip() for text in texts]))
    }

    if tokenizer is not None:
        process = lambda texts: chunk_tokens(
            tokenize_texts(
                list(filter(None, [text.strip() for text in texts])),
                tokenizer,
                add_special_tokens=False,
            ),
            tokenizer.model_max_length,
        )

    dsd = dsd.map(
        process,
        batched=True,
        input_columns="text",
        remove_columns=dsd["train"].column_names,
        num_proc=num_proc,
    )
    print(dsd)
    dsd = dsd["train"].train_test_split(test_size=0.1, seed=42)
    dsd_valid_test = dsd["test"].train_test_split(test_size=0.5, seed=42)
    dsd = DatasetDict(
        {
            "train": dsd["train"],
            "validation": dsd_valid_test["train"],
            "test": dsd_valid_test["test"],
        }
    )
    return dsd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--np",
        type=int,
        default=os.cpu_count(),
        metavar=os.cpu_count(),
        help="number of process to do pre-tokenization",
    )
    parser.add_argument(
        "--np-to-parquet",
        type=int,
        default=os.cpu_count() // 2,
        metavar=os.cpu_count() // 2,
        help="number of process to write parquet files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scr/cache",
        metavar="/scr/cache",
        help="cache dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mixed",
        metavar="data/mixed",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Specify the path if you want to do pre-tokenization",
    )
    parser.add_argument(
        "--ignore-verifications",
        action="store_true",
        help="Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/…).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dsd_list = [
        preprocess_range3_wiki40b_ja(
            num_proc=args.np,
            cache_dir=str(cache_dir),
            tokenizer=tokenizer,
            ignore_verifications=args.ignore_verifications,
        ),
        preprocess_range3_narou20(
            num_proc=args.np,
            cache_dir=str(cache_dir),
            tokenizer=tokenizer,
            ignore_verifications=args.ignore_verifications,
        ),
        # preprocess_range3_cc100_ja(
        #     num_proc=args.np,
        #     cache_dir=str(cache_dir),
        #     tokenizer=tokenizer,
        #     ignore_verifications=args.ignore_verifications,
        # ),
    ]

    # concatenete datasets
    dsd = DatasetDict(
        {
            split: concatenate_datasets([dsd[split] for dsd in dsd_list])
            for split in ["train", "validation", "test"]
        }
    ).shuffle(seed=42)

    # shard dataset
    for split, num_shards in [("train", 48), ("validation", 4), ("test", 4)]:
        ds = dsd[split]
        with Pool(processes=args.np_to_parquet) as p:
            p.starmap(
                save_to_parquet,
                zip(
                    [output_dir] * num_shards,
                    [ds] * num_shards,
                    [split] * num_shards,
                    [num_shards] * num_shards,
                    range(num_shards),
                ),
            )


def save_to_parquet(
    output_dir: Path,
    ds: Dataset,
    split: str,
    num_shards: int,
    index: int,
):
    ds.shard(
        num_shards=num_shards,
        index=index,
        contiguous=True,
    ).to_parquet(f"{output_dir}/{split}_{index:05}_of_{num_shards:05}.parquet")


if __name__ == "__main__":
    main()
