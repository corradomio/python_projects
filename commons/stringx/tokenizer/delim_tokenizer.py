from .regexp_tokenizer import RegExpTokenizer

RE_SPECIAL_CHARS = ".^$*+?\\[]|(){}"


def _escape_char(chars):
    n = len(chars)
    for i in range(n):
        if chars[i] in RE_SPECIAL_CHARS:
            chars[i] = "\\" + chars[i]
    return chars


class DelimTokenizer(RegExpTokenizer):

    def __init__(self, delim=" .,;:!?()[]{}", unique=False):
        if isinstance(delim, str):
            delim = list(iter(delim))
        delim = _escape_char(delim)
        sep = "|".join(delim)
        super().__init__(sep, unique=unique)

    def tokenize(self, s: str) -> list[str]:
        parts = self._re.split(s)
        # remove empty string
        parts = list(filter(lambda s: len(s) > 0, parts))

        return self.as_set(parts)
