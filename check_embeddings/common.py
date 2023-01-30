import logging.config
import tomotopy as tp
import os
from path import Path as path

NAME = "cocome"
ROOT_DIR = f"D:/Dropbox/Software/Mathematica/GraphNLP/{NAME}"
FILE_IDS = f"D:/Dropbox/Software/Mathematica/GraphNLP/{NAME}/graph-source-vertices.csv"
DOT_TOKENS = ".tokens"

MIN_LENGTH = 4
JAVA_KEYWORDS = set([
    'abstract',
    'assert',
    'boolean',
    'break'
    'byte',
    'case',
    'catch',
    'char',
    'class',
    'const',
    'continue',
    'default',
    'do',
    'double',
    'else',
    'enum',
    'extends',
    'final',
    'finally',
    'float',
    'for',
    'goto',
    'if',
    'implements',
    'import',
    'instanceof',
    'int',
    'interface',
    'long',
    'native',
    'new',
    'package',
    'private',
    'protected',
    'public',
    'return',
    'short',
    'static',
    'strictfp',
    'super',
    'switch',
    'synchronized',
    'this',
    'thorw',
    'throws',
    'transient',
    'try',
    'void',
    'volatile',
    'while',

    'record',
    'exports',
    'module',
    'non-sealed',
    'open',
    'opens',
    'permits',
    'provides',
    'requires',
    'sealed',
    'to',
    'transitive',
    'uses',
    'var',
    'with',
    'yield',

    'true',
    'false',
    'null',

    'override',
    'deprecated',
    'supress',
    'warning',
    'not',

    'string',
    'list', 'array', 'hash', 'map',
    'iterator', 'iterable',
    'exception'
])


def load_fileids():
    filed2id = dict()
    with open(FILE_IDS) as rdr:
        # skip 'id, s, name'
        next(rdr)
        for line in rdr:
            parts = line.split(",")
            id = parts[0]
            # note: the path contains '"' !!!
            path = parts[2][1:-2]
            filed2id[path] = id
    return filed2id


def load_doc(p: path, minlen=0, skipwords=set()):
    tokens = []
    with open(p, encoding="iso-8859-1") as file:
        for line in file:
            tokens.append(line.strip())

    tokens = list(filter(lambda t: len(t) >= minlen and t not in skipwords, tokens))
    # tokens = list(filter(lambda t: isinstance(t, str), tokens))
    return tokens


def relative_path(p):
    s = len(ROOT_DIR)+1     # skip the last '/'
    e = len(DOT_TOKENS)
    path = str(p).replace('\\', '/')
    rpath = path[s:-e]
    return rpath
# end


def load_corpus(minlen=0, skipwords=set()):
    corpus = []
    files = []
    root = path(ROOT_DIR)
    for p in root.walkfiles("*.tokens"):
        if p.name.startswith('counts'):
            continue
        files.append(relative_path(p))
        corpus.append(load_doc(p, minlen=minlen, skipwords=skipwords))
    return corpus, files
# end

