import re

from .tokenizer import Tokenizer


class SplitTokenizer(Tokenizer):

    def __init__(self, sep=None, unique=False):
        super().__init__(unique=unique)
        self.sep = sep

    def tokenize(self, s: str) -> list[str]:
        parts = s.split(self.sep)

        return self._as_set(parts)
