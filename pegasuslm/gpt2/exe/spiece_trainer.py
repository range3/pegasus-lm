import os
import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from pegasuslm.utils import pretty_fmt
from pegasuslm import PegasusGPT2Tokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="train vocab model using SentencePiece"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model",
        metavar="model",
        help="output dir",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default=None,
        # default="data/narou_small",
        metavar="data/narou_small",
        help="input dataset name or path",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default=None,
        # default="data/mixed.txt",
        metavar="data/mixed.txt",
        help="input text file path",
    )
    parser.add_argument(
        "--ignore-verifications",
        action="store_true",
        help="Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/…).",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="spiece",
        metavar="spiece",
        help="prefix of model and vocab files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        metavar="50000",
        help="size of sentences the trainer loads",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=1000000,
        metavar="1000000",
        help="maximum size of sentences the trainer loads",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=os.cpu_count(),
        metavar=os.cpu_count(),
        help="number of process to do preprocessing and training",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
    )
    args = parser.parse_args()
    if args.disable_progress_bar:
        disable_progress_bar()

    if (args.input_text and args.input_dataset) or (
        not args.input_text and not args.input_text
    ):
        raise ValueError("Please specify one of --input-text or --input-dataset")

    output_dir = Path(args.output_dir).resolve()
    model_dir = (
        output_dir
        / f'{pretty_fmt(args.vocab_size, suffix="V")}_{pretty_fmt(args.input_sentence_size, suffix="S")}'
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = model_dir / f"{args.model_prefix}"

    if args.input_dataset:
        # preprocessing
        ds = load_dataset(
            args.input_dataset,
            split="train",
            ignore_verifications=args.ignore_verifications,
            num_proc=args.np,
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

        # train sentencepiece model
        tokenizer = PegasusGPT2Tokenizer.train_from_dataset(
            ds,
            "text",
            model_prefix=str(model_prefix),
            vocab_size=args.vocab_size,
            input_sentence_size=args.input_sentence_size,
            num_threads=args.np,
        )
    else:
        # train sentencepiece model
        tokenizer = PegasusGPT2Tokenizer.train_from_textfile(
            args.input_text,
            model_prefix=str(model_prefix),
            vocab_size=args.vocab_size,
            input_sentence_size=args.input_sentence_size,
            num_threads=args.np,
        )

    tokenizer.save_pretrained(str(model_dir))


if __name__ == "__main__":
    main()
