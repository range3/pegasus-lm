import os
import re
import warnings
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Iterator
from typing_extensions import Self

from transformers import PreTrainedTokenizer, AutoTokenizer
import neologdn
import sentencepiece as spm
from prefetch_generator import prefetch
from .config import PegasusGPT2Config

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class PegasusGPT2Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        lf_token="<n>",
        sep_token="[SEP]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        do_lower_case=False,
        keep_accents=False,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            lf_token=lf_token,
            do_lower_case=do_lower_case,
            keep_accents=keep_accents,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
        self.lf_token = lf_token
        self.do_lower_case = do_lower_case
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    re_newline = re.compile("\r?\n")

    @classmethod
    def normalize(cls, text: str, lf_token: str = "<n>"):
        return neologdn.normalize(
            cls.re_newline.sub(lf_token, text.strip()), tilde="normalize_zenkaku"
        )

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        text = PegasusGPT2Tokenizer.normalize(text)
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        if token != self.lf_token:
            return token
        else:
            return "\n"

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:
        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    @classmethod
    def train_from_dataset(
        cls,
        dataset,
        column: str,
        max_prefetch: int = 3,
        **kwargs,
    ) -> Self:
        # @prefetch(max_prefetch=max_prefetch)
        def sentence_feeder():
            for row in dataset:
                yield row[column]

        return cls.train_from_iterator(sentence_feeder(), **kwargs)

    @classmethod
    def spiece_train_kwargs(
        cls,
        input_sentence_size: int = 1000,
        max_sentence_length: int = 65536,
        vocab_size: int = 50000,
        num_threads: int = os.cpu_count(),
        model_prefix: str = "spiece",
    ) -> Dict[str, Any]:
        return {
            "num_threads": num_threads,
            "train_extremely_large_corpus": True,
            "max_sentence_length": max_sentence_length,
            "input_sentence_size": input_sentence_size,
            "model_prefix": model_prefix,
            "vocab_size": vocab_size,
            "character_coverage": 0.9995,
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "control_symbols": [
                "[SEP]",
                "[CLS]",
                "[MASK]",
            ],
            "user_defined_symbols": ["<n>"],
            "add_dummy_prefix": False,
            "shuffle_input_sentence": False,
        }

    @classmethod
    def train_from_iterator(
        cls,
        iterator: Iterator[str],
        tokenizer_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> Self:
        train_kwargs = cls.spiece_train_kwargs(**kwargs)
        spm.SentencePieceTrainer.Train(
            sentence_iterator=iterator,
            **train_kwargs,
        )
        return PegasusGPT2Tokenizer(
            f"{train_kwargs['model_prefix']}.model",
            **tokenizer_kwargs,
        )

    @classmethod
    def train_from_textfile(
        cls,
        text_file: str,
        tokenizer_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> Self:
        train_kwargs = cls.spiece_train_kwargs(**kwargs)
        spm.SentencePieceTrainer.Train(
            input=text_file,
            **train_kwargs,
        )
        return PegasusGPT2Tokenizer(
            f"{train_kwargs['model_prefix']}.model",
            **tokenizer_kwargs,
        )


AutoTokenizer.register(PegasusGPT2Config, PegasusGPT2Tokenizer)
