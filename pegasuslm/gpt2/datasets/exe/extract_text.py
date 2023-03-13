import os
import argparse
from pathlib import Path
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from pegasuslm import PegasusGPT2Tokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scr/cache",
        metavar="/scr/cache",
        help="cache dir",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/mixed",
        metavar="data/mixed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mixed.txt",
        metavar="data/mixed.txt",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=os.cpu_count(),
        metavar=os.cpu_count(),
        help="number of process to do preprocessing and training",
    )
    parser.add_argument(
        "--ignore-verifications",
        action="store_true",
        help="Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/…).",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
    )
    args = parser.parse_args()
    if args.disable_progress_bar:
        disable_progress_bar()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        args.input,
        split="train",
        ignore_verifications=args.ignore_verifications,
        num_proc=args.np,
        cache_dir=args.cache_dir,
    )
    print(ds)
    ds = ds.map(
        lambda batch: {
            "text": [PegasusGPT2Tokenizer.normalize(text) for text in batch],
        },
        batched=True,
        input_columns="text",
        remove_columns=[c for c in ds.column_names if c != "text"],
        num_proc=args.np,
    )

    with output.open(mode="w+") as f:
        r = range(len(ds))
        if not args.disable_progress_bar:
            r = tqdm(r)
        for i in r:
            f.write(ds[i]["text"] + "\n")


if __name__ == "__main__":
    main()
