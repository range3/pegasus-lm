#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 range3 ( https://github.com/range3 )
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from pegasuslm import (
    PegasusGPT2Config,
    PegasusGPT2Tokenizer,
    PegasusGPT2Model,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument(
        "--stop_token",
        type=str,
        default="</s>",
        help="Token at which text generation is stopped",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--p", type=float, default=0.95)

    parser.add_argument(
        "--prefix", type=str, default="", help="Text added prior to input."
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}"
    )

    set_seed(args)

    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    logger.info(args)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    prefix = args.prefix if args.prefix else ""
    # encoded_prompt = tokenizer.encode(
    encoded_prompt = tokenizer(
        prefix + prompt_text,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    encoded_prompt = encoded_prompt.to(args.device)
    print(encoded_prompt)

    output_sequences = model.generate(
        **encoded_prompt,
        max_length=args.length + len(encoded_prompt["input_ids"][0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
        bad_words_ids=[[tokenizer.unk_token_id]],
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text
            + text[
                len(
                    tokenizer.decode(
                        encoded_prompt["input_ids"][0],
                        clean_up_tokenization_spaces=True,
                    )
                ) :
            ]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)

    return generated_sequences


if __name__ == "__main__":
    main()
