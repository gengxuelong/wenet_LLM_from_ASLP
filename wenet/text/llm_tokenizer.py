from os import PathLike
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer

from wenet.text.base_tokenizer import BaseTokenizer

from wenet.utils.file_utils import read_non_lang_symbols


class LmmTokenizer(BaseTokenizer):

    def __init__(
            self,
            llm_path: str,
            *args,
            **kwargs,
    ) -> None:
        self.llm_path = llm_path
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)  # for Atom
        # ids = llama_tokenizer(sample['txt'], return_tensors="pt").input_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        recovery = {'tokenizer': None}
        self.__dict__.update(recovery)

    def _build_tiktoken(self):
        if self.tokenizer is None:
            from whisper.tokenizer import get_tokenizer
            self.tokenizer = get_tokenizer(multilingual=self.multilingual,
                                           num_languages=self.num_languages,
                                           language=self.language,
                                           task=self.task)
            self.t2i = {}
            self.i2t = {}
            for i in range(self.tokenizer.encoding.n_vocab):
                unit = str(
                    self.tokenizer.encoding.decode_single_token_bytes(i))
                if len(unit) == 0:
                    unit = str(i)
                unit = unit.replace(" ", "<space>")
                # unit = bytes(unit, 'utf-8')
                self.t2i[unit] = i
                self.i2t[i] = unit
            assert len(self.t2i) == len(self.i2t)

    def tokenize(self, line: str) -> Tuple[List[str], List[int]]:
        ids = self.tokenizer(line, return_tensors="pt").input_ids
        tokens = self.tokenizer.tokenize(line)
        return [""], ids

    def detokenize(self, ids: List[int]) -> Tuple[str, List[str]]:
        self._build_tiktoken()
        tokens = [self.i2t[d] for d in ids]
        text = self.tokenizer.encoding.decode(ids)
        return text, tokens

    def text2tokens(self, line: str) -> List[str]:
        self._build_tiktoken()
        return self.tokenize(line)[0]

    def tokens2text(self, tokens: List[str]) -> str:
        ids = self.tokens2ids(tokens)
        return ""

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def ids2tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def symbol_table(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()
