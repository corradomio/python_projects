import re
from .tokenizer import Tokenizer

# Special characters
#   . ^ $ * + ? \ [ ] | ( )
#
#   *?, +?, ??
#   *+, ++, ?+
#   {m} {m,n} {m,n}? {m,n}+
#   (?...)  (?aiLmsux)    (?:...)   (?aiLmsux-imsx:...)
#   (?>...) (?P<name>...) (?P=name)
#   (?#...) (?!...) (?<=...) (?<!...)
#   (?(id/name)yes-pattern|no-pattern)
#
#   \number
#   \A \b \B \d \D \s \S \w \W \Z
#
#   \a      \b      \f      \n
#   \N      \r      \t      \u
#   \U      \v      \x      \\


class RegExpTokenizer(Tokenizer):

    def __init__(self, regexp, unique=False):
        super().__init__(unique=unique)
        self.re = regexp
        self._re = re.compile(regexp)

    def tokenize(self, s: str) -> list[str]:
        parts = list(filter(None, self._re.findall(s)))

        return self.as_set(parts)


class AlphabeticTokenizer(RegExpTokenizer):

    def __init__(self, unique=False):
        super().__init__('[a-zA-Z]+', unique=unique)


class AlphanumericTokenizer(RegExpTokenizer):

    def __init__(self, unique=False):
        super().__init__('[a-zA-Z0-9]+', unique=unique)
