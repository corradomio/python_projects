import struct as st
from collections import defaultdict
from datetime import datetime
from random import shuffle
from typing import Dict, List


def shingles(embedding: List[float]):
    sh = set()
    for f in embedding:
        # bytes = st.unpack('8B', st.pack('d', f))
        bytes = st.unpack('4H', st.pack('d', f))
        # bytes = st.unpack('2I', st.pack('d', f))
        sh.update(bytes)
    return sh
# end


def min_hash(enc1, enc2):
    n = len(enc1)
    l = list(range(n))
    shuffle(l)
    pass
# end


class Shingles:

    def __init__(self):
        self.allsh: Dict[int, int] = dict()
        self.lensh: Dict[int, int] = defaultdict(lambda: 0)
        self.n = 0

    def shingles(self, embedding: List[float]):
        sh = shingles(embedding)
        self.update(sh)
        return sh

    def encode(self, embedding: List[float]):
        sh = shingles(embedding)
        enc = [0]*self.n
        for s in sh:
            i = self.allsh[s]
            enc[i] = 1
        return enc

    def update(self, sh):
        for s in sh:
            if s not in self.allsh:
                self.allsh[s] = self.n
                self.n += 1

        lsh = 50 * (len(sh) // 50)
        self.lensh[lsh] += 1
    # end

    # def get(self, s):
    #     if s not in self.allsh:
    #         self.allsh[s] = self.n
    #         self.n += 1
    #     return self.allsh[s]

    # def as_shingles(self, embedding):
    #     sh = set()
    #     for f in embedding:
    #         # bytes = st.unpack('8B', st.pack('d', f))
    #         # bytes = st.unpack('4H', st.pack('d', f))
    #         bytes = st.unpack('2I', st.pack('d', f))
    #         sh.update(bytes)
    #     return sh
    # # end

    def compare(self, e1, e2):
        enc1 = self.encode(e1)
        enc2 = self.encode(e2)
        cmp = min_hash(enc1, enc2)
        pass

    def show(self):
        print("allsh:  ", len(self.allsh))
        for k in sorted(self.lensh.keys()):
            v = self.lensh[k]
            print(f"{k:5}: {v:6}")
    # end
# end


class WordEmbeddings:

    def __init__(self):
        self.embeddings: Dict[str, List[float]] = dict()

    def __setitem__(self, word, embedding):
        self.embeddings[word] = embedding

    def __getitem__(self, word):
        return self.embeddings[word]
# end


def load_data(shdict, embeddings, maxc = 0):
    print(f"start ...")
    start = datetime.now()
    lastt = start
    lastc = 0
    count = 0

    with open("E:\\Datasets\\fastText\\crawl-300d-2M-subword.vec", encoding="iso8859") as fastText:
        next(fastText)
        for line in fastText:
            parts = line.split(' ')
            word = parts[0]
            embedding = list(map(float, parts[1:]))

            shdict.shingles(embedding)
            embeddings[word] = embedding

            # if len(sh) > lensh:
            #     lensh = len(sh)
            #     print(word, ":", lensh, "->",  sh)
            # print(word)
            count += 1

            if count % 1000 == 0 and ((datetime.now() - lastt).seconds > 3.):
                print(f"... {count:7} + {count - lastc:5}")
                shdict.show()
                lastt = datetime.now()
                lastc = count
            # end

            if 0 < maxc < count:
                break
        # end
    # end

    shdict.show()

    delta = (datetime.now() - start).total_seconds()/60
    print(f"end in {delta} m")
# end


def main():
    shdict = Shingles()
    embeddings = WordEmbeddings()

    load_data(shdict, embeddings, 10000)

    man = embeddings["man"]
    woman = embeddings["woman"]
    king = embeddings["king"]
    queen = embeddings["queen"]

    # print(shdict.encode(man))
    # print(shdict.encode(woman))
    # print(shdict.encode(king))
    # print(shdict.encode(queen))

    print(shdict.compare(man, woman))
    print(shdict.compare(king, queen))

    pass
# end


if __name__ == "__main__":
    main()
