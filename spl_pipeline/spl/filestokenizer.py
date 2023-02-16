import re
import math
from collections import Counter
from random import shuffle

from .commons import *
from .lang_keywords import LANGUAGE_KEYWORDS
from .loggingx import Logger
from .utils_stdlib import flatten, list_filter, set_filter, list_map


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_ident(s: str) -> bool:
    """
    Check if s ia a valid 'identifier' as defined in Java or other similar programming
    languages

    :param s: string to test
    :return: false if it is not a valid identifier, True otherwise
    """
    if len(s) == 0:
        return False
    c = s[0]
    return 'A' <= c <= 'Z' or 'a' <= c <= 'z' or c in "@_$"
# end


def stem_ident(s: str) -> str:
    """
    Remove last 's' (mini stemming)
    :param s:
    :return:
    """
    if len(s) < 3:
        return s
    if s.endswith('ss'):
        return s
    elif s.endswith('is'):
        return s
    # elif s.endswith('ies'):
    #     return s[0:-3] + "y"
    # elif s.endswith('ses'):
    #     return s[0:-2]
    # elif s.endswith('hes'):
    #     return s[0:-2]
    # elif s.endswith('oes'):
    #     return s[0:-2]
    # elif s.endswith('xes'):
    #     return s[0:-2]
    # elif s.endswith('zes'):
    #     return s[0:-2]
    elif s.endswith('s'):
        return s[0:-1]
    else:
        return s

    # stemming library
    # return stemming.porter.stem(s)
# end


def shuffle_list(l: list) -> list:
    """
    Shuffles the list and return itself.
    Note: 'random.shuffle(l)' shuffles the list in-place and returns None
    :param l: list to shuffle
    :return: the list itself
    """
    shuffle(l)
    return l
# end


# ---------------------------------------------------------------------------
# TokenDict/TokenInfo
# ---------------------------------------------------------------------------

T = ""


class TokenInfo:
    def __init__(self, token=''):
        self.token: str = token
        self.count: int = 0
        self.docs: int = 0

    def __repr__(self):
        return f"d:{self.docs}, c:{self.count}"
# end


class TokenDict:

    def __init__(self):
        self.tdict: dict[str, TokenInfo] = dict()
        self.tdict[T] = TokenInfo('')

    def add_all(self, tokens):
        self.tdict[T].docs += 1
        ctoks = set()
        for token in tokens:
            assert len(token) > 0

            if token not in self.tdict:
                self.tdict[token] = TokenInfo(token)

            tinfo = self.tdict[token]
            if token not in ctoks:
                tinfo.docs += 1
                ctoks.add(token)
            # end
            tinfo.count += 1
            self.tdict[T].count += 1
        # end
    # end

    def keys(self):
        return self.tdict.keys()

    def __getitem__(self, token):
        return self.tdict[token]

    def __len__(self):
        return len(self.tdict)

    def tfidf(self, token):
        if token not in self.tdict:
            return 0.

        all = self.tdict[T]
        tok = self.tdict[token]

        return tok.count/all.count * math.log(all.docs/tok.docs)
    # end

    def tokens(self, min_len=0, min_count=0, min_docs=0, min_tfidf=0.) -> set[str]:
        if min_len == 0 and min_count == 0 and min_docs == 0 and min_tfidf == 0:
            return set(self.tdict.keys())

        return set_filter(lambda t: len(t) >= min_len
                                    and self.tdict[t].docs >= min_docs
                                    and self.tdict[t].count >= min_count
                                    and self.tfidf(t) >= min_tfidf,
                          self.tdict.keys())
    # end
# end


# ---------------------------------------------------------------------------
# FilesTokenizer
# ---------------------------------------------------------------------------

class FileInfo:
    def __init__(self, index: int, f):
        # finfo = {
        #     "index": len(self._files),
        #     "file": str(f),
        #     "name": f.name,
        #     "": f.stem
        # }
        self.index = index
        self.file = str(f)
        self.name = f.name
        self.stem = f.stem
    # end

    def __getitem__(self, item):
        if item == 'index':
            return self.index
        if item == 'file':
            return self.file
        if item == 'name':
            return self.name
        if item == 'stem':
            return self.stem
        else:
            raise ValueError(f"Unknown key {item}")

    def __repr__(self):
        return self.name
# end


class FilesTokenizer:
    def __init__(self,
                 tokenize: Union[None, bool, str] = True,
                 camel_case: bool = True,
                 lower: bool = True,
                 word_only: bool = True,
                 stem: bool = False,
                 unique: bool = False,
                 shuffle: bool = False,
                 stopwords: Union[None, str, set[str], list[str]] = None,
                 min_len: int = 0,
                 min_docs: int = 0,
                 min_count: int = 0,
                 min_tfidf: float = 0.):
        """

        :param tokenize: it to tokenize the text.
                         Otherwise, the file content is collected as simple string
        :param camel_case: if to split strings written in camel case
        :param lower: if to convert all tokens in lowercase
        :param word_only: if to keep only tokens starting with a letter [a-z]
        :param stem: if to stem all tokens.
        :param unique: if to remove all duplicates
        :param shuffle: if to shuffle the tokens
        :param stopwords: None, or a programming language name, or a list/set of stopwords
        :param min_len: minimum token len to keep
        :param min_docs: minimum number of documents
        """
        # tokenizer -> re
        if tokenize is None:
            self.re = None
        elif isinstance(tokenize, bool):
            if tokenize:
                self.re = "[ \t\n\r%#&@$,.:;?!*/0-9'\"()\\[\\]\\\\/{}=<>_+-]"
            else:
                self.re = None
        elif isinstance(tokenize, str):
            self.re = tokenize
        else:
            raise ValueError(f"Invalid 'tokenize' parameter: '{tokenize}'")

        self.camel_case = camel_case
        self.lower = lower
        self.ident_only = word_only
        self.stem = stem
        self.unique = unique
        self.shuffle = shuffle

        # stopwords: is strin
        if stopwords is None:
            self.stopwords = set()
        elif isinstance(stopwords, str):
            self.stopwords = LANGUAGE_KEYWORDS[stopwords]
        else:
            self.stopwords = set(stopwords)

        self.min_len = min_len
        self.min_docs = min_docs
        self.min_count = min_count
        self.min_tfidf = min_tfidf

        self._files = list()
        self._used_tokens = set()
        self._corpus = []
        self._tdict = TokenDict()
        self._log = Logger.getLogger("FilesTokenizer")
    # end

    @property
    def files(self) -> list[FileInfo]:
        """
        Each element of the list is a dictionary with keys:

            'index': file index, based on the reading order
            'file':  fully qualified path
            'path':  relative path respect th project home
            'name':  file name with extension
            'stem':  file name without extension

        Note:

        :return: list of processed files
        """
        # check the order
        n = len(self._files)
        for i in range(n):
            assert self._files[i].index == i

        return self._files

    @property
    def paths(self) -> list[str]:
        return list_map(lambda fi: fi['path'], self._files)

    @property
    def names(self) -> list[str]:
        return list_map(lambda fi: fi['stem'], self._files)

    @property
    def corpus(self) -> Union[list[str], list[list[str]]]:
        """
        List of strings (content of each file) or list of list of tokens
        :return:
        """
        # # shuffle the tokens at each call
        # if self.shuffle and self.re is not None:
        #     self._corpus = list(map(shuffle_list, self._corpus))
        # return self._corpus
        return self.get_corpus(bag=False)

    @property
    def documents(self):
        """Alias for 'corpus'"""
        return self.corpus

    @property
    def dictionary(self) -> TokenDict:
        """
        Dictionary 'token' -> Info(docs=n_documents, count=n_occurrences)
        The token '' (empty string) is used as placeholder to save the
        total number of documents and the total number of tokens read.

        :return: dictionary  'token' -> Info(docs=n_documents, count=n_occurrences)
        """
        return self._tdict

    def fit(self, files: Iterable[Union[str, Path]]):
        self._log.info("fit files ...")
        self._corpus = []

        for f in files:
            tokens = self._get_tokens(f)
            # special case if no tokenizer is used
            if tokens is None: continue

            # corpus
            self._corpus.append(tokens)

            # tokens
            self._tdict.add_all(tokens)
        # end

        # filter used tokens
        self._filter_tokens()

        # check if all documents are NOT empty
        self._check_not_empty()

        # scan complete
        tinfo: TokenInfo = self._tdict[T]
        self._log.info(f"done [{tinfo.docs}/{tinfo.count}/{len(self._used_tokens)}]")
        return self
    # end

    def _get_tokens(self, f) -> Union[None, list[str]]:
        # info on the file
        finfo = FileInfo(len(self._files), f)
        self._files.append(finfo)

        # n of documents processed
        # NOT NECESSARY
        # self._tdict[T].docs += 1

        # file -> text
        text = read_text(f)

        # if it is not tokenized, add it as text
        if self.re is None or not self.re:
            self._corpus.append(text)
            return None

        # tokenize
        tokens = re.split(self.re, text)
        # CamelCase
        if self.camel_case:
            tokens = flatten(split_camel(t) for t in tokens)
        # lowerCase
        if self.lower:
            tokens = [t.lower() for t in tokens]
        # identifies
        if self.ident_only:
            tokens = list_filter(is_ident, tokens)
        # stemming
        if self.stem:
            tokens = list_map(stem_ident, tokens)
        # stopwords
        if self.stopwords:
            tokens = list_filter(lambda t: t not in self.stopwords, tokens)
        # unique
        if self.unique:
            tokens = list(set(tokens))
        # min length
        if self.min_len > 0:
            min_len = self.min_len
            tokens = list_filter(lambda t: len(t) >= min_len, tokens)

        return tokens
    # end

    def _filter_tokens(self):
        self._used_tokens = self._tdict.tokens()

        if self.min_docs == 0 and self.min_count == 0 or self.min_len == 0 and self.min_tfidf == 0:
            return

        tvalid = self._tdict.tokens(
            min_docs=self.min_docs,
            min_count=self.min_count,
            min_len=self.min_len,
            min_tfidf=self.min_tfidf
        )

        self._used_tokens = tvalid

        # filter the documents based on the valid tokens
        n = len(self._corpus)
        for i in range(n):
            doc = self._corpus[i]
            doc = list_filter(lambda t: t in tvalid, doc)
            self._corpus[i] = doc
        # end
    # end

    def _check_not_empty(self):
        n = len(self._corpus)
        n_empty = 0
        for i in range(n):
            if len(self._corpus[i]) > 0:
                continue
            self._corpus[i] = ["empty"]
            n_empty += 1
        # end
        if n_empty > 0:
            self._log.warn(f'... found {n_empty} empty documents')
    # end

    # -----------------------------------------------------------------------

    def get_corpus(self, bag=False, normalized=False):
        """
                List of strings (content of each file) or list of list of tokens
                :return:
                """
        # shuffle the tokens at each call
        if self.shuffle and self.re is not None:
            self._corpus = list_map(shuffle_list, self._corpus)
        if not bag:
            return self._corpus

        def normalize(bag: dict):
            maxv = max(bag.values())
            for key in bag:
                bag[key] = bag[key]/maxv
            return bag

        bag_corpus = list_map(Counter, self._corpus)
        if normalized:
            bag_corpus = list_map(normalize, bag_corpus)
        return bag_corpus
    # end

    # -----------------------------------------------------------------------

    def save(self, file: str):
        file = file.replace('%s', 'dictionary')
        file = file.replace('%1s', '')
        self._log.info(f"saving '{file}'")

        with open(file, mode='w', encoding='utf-8') as wrt:
            wrt.write('token,docs,count,tfidf\n')
            tinfo = self._tdict[T]
            wrt.write(f"{tinfo.token},{tinfo.docs},{tinfo.count},0.0\n")

            for token in self._tdict.keys():
                if token == '': continue
                tinfo = self._tdict[token]
                tfidf = self._tdict.tfidf(token)
                try:
                    wrt.write(f"{token},{tinfo.docs},{tinfo.count},{tfidf:.6}\n")
                except Exception as e:
                    self._log.error(f"Unable to save {token}")
            # end
        # end
        pass
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
