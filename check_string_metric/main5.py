# from abydos.distance import Tversky
from stringx.similarity import Jaccard, Tversky

#
# .sim( a, b)
# .dist(a, b)
#


def main():
    d = Jaccard()
    print(d.similarity("ciao", "ciccio"))
    d = Tversky(alpha=0, beta=0)
    print(d.similarity("ciao", "ciccio"))
    pass


if __name__ == "__main__":
    main()