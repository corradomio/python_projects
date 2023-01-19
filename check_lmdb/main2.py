from caches import CacheManager, InMemoryCaches
from timing import tprint


class GloVe:

    def __init__(self, path, caches: CacheManager):
        self.path = path
        self.dict = caches.cache("glove", str, list[int])
    # end

    def open(self):
        tprint(f"Loading {self.path} ...")
        with open(self.path, encoding="utf8") as rdr:
            count = 0
            for line in rdr:
                line = line.strip()
                parts = line.split(" ")
                word = parts[0]
                embedding = list(map(float, parts[1:]))
                self.dict[word] = embedding
                count += 1
            # end
        # end
        tprint(f"Loaded {count} words")
    # end

    def __getitem__(self, item):
        return self.dict[item]

    def __contains__(self, item):
        return item in self.dict

    def __iter__(self):
        return self.dict.__iter__()

    def close(self):
        pass


def main():
    caches = InMemoryCaches.open("scitools")

    glove = GloVe("D:/Datasets/Embeddings/glove.6B.50d.txt", caches)
    glove.open()

    print(glove["man"])
    print(glove["woman"])
    print("children" in glove)

    count = 0
    for w in glove:
        print(w)
        count += 1
        if count > 100: break

    pass


if __name__ == "__main__":
    main()
