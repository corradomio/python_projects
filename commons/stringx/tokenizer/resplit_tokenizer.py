from .regexp_tokenizer import RegExpTokenizer


class RESplitTokenizer(RegExpTokenizer):

    def __init__(self, sep=None, unique=False):
        super().__init__(sep, unique=unique)

    def tokenize(self, s: str) -> list[str]:
        parts = self._re.split(s)

        return self.as_set(parts)
