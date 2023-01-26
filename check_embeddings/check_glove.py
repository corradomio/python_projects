import logging.config
import math

import numpy as np
from common import *

GLOVE_EMB = "D:/Datasets/GloVe/glove.6B.50d.txt"

class Info:
    docs = 0
    count = 0


def words_statistics(corpus, unique=False, wlen=0):
    wstat = dict()
    wstat[""] = [0, 0]
    for doc in corpus:
        if unique:
            doc = set(doc)
        if wlen > 0:
            doc = list(filter(lambda w: len(w) >= wlen, doc))
        wstat[""][1] += 1
        wdoc = set()
        for w in doc:
            if w not in wstat:
                wstat[w] = [0, 0]
            if w not in wdoc:
                wstat[w][1] += 1
                wdoc.add(w)
            wstat[w][0] += 1
            wstat[""][0] += 1
        # end
    # end
    count, docs = wstat[""]
    logging.info(f"end {count}/{docs}")
    return wstat


class WordEmbedding:
    def __init__(self):
        self.wd = dict()
        self.refw = None

    def __setitem__(self, word, vec):
        self.wd[word] = vec
        self.refw = word

    def __getitem__(self, word):
        if word in self.wd:
            return self.wd[word]
        else:
            return np.zeros_like(self.wd[self.refw])
    # end
# end


def load_glove():
    logging.info("Loading GloVe ...")
    emb = WordEmbedding()
    with open(GLOVE_EMB, encoding='utf-8') as rdr:
        for line in rdr:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = list(map(float, parts[1:]))
            vec = np.array(vec)
            emb[word] = vec
    # end
    logging.info("done")
    return emb


def tfidf(wstat, w):
    gc, gd = wstat[""]
    wc, wd = wstat[w]
    return wc/gc * math.log(gd/wd)


def doc_embedding(emb, wstats, doc, unique=False):
    if unique:
        doc = set(doc)

    dvec = np.zeros_like(emb["the"])
    for w in doc:
        weight = tfidf(wstats, w)
        dvec += weight * emb[w]

    return dvec
# end


def save_embeddings(unique=False):
    corpus, files = load_corpus(minlen=MIN_LENGTH, skipwords=JAVA_KEYWORDS)
    n = len(files)
    wstats = words_statistics(corpus)
    fileids = load_fileids()
    emb = load_glove()

    fname = "cocome-glove-1.vec" if unique else "cocome-glove-n.vec"

    with open(fname, mode='w') as wrt:
        for i in range(n):
            file = files[i]
            if file not in fileids:
                print(f"file skipped: '{file}'")
                continue

            doc = corpus[i]
            dvect = doc_embedding(emb, wstats, doc, unique)

            svect = list(map(str, dvect))
            semb = ",".join(svect)

            fileid = fileids[file]
            wrt.write(f"{fileid},{semb}\n")
    # end
# end

def main():
    save_embeddings(False)
    save_embeddings(True)


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

