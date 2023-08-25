from .tokenizer import Tokenizer


class QGramTokenizer(Tokenizer):

    def __init__(self, k=2, padding='#', pad=True, unique=False):
        super().__init__(unique=unique)
        self.k = k
        self.padding = padding
        self.pad = pad

    def tokenize(self, s: str) -> list[str]:
        k = self.k
        s = self._apply_pad(s)
        n = len(s)

        parts = []
        for i in range(0, n-k+1):
            part = s[i:i+k]
            parts.append(part)

        return self._as_set(parts)

    def _apply_pad(self, s):
        if self.pad:
            p = self.k - 1
            s = self.padding*p + s + self.padding*p
        return s