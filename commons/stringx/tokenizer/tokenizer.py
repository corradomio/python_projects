class Tokenizer:

    def __init__(self, unique=False):
        self.unique = unique

    def tokenize(self, s: str) -> list[str]:
        ...

    def as_set(self, parts):
        if self.unique:
            parts = list(set(parts))
        else:
            # parts = sorted(parts)
            pass
        return parts
