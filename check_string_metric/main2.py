from stringx.similarity import *
from py_stringmatching.similarity_measure.hamming_distance import HammingDistance


def main():
    print(Hamming().distance("ciao", "ciro"))
    print(HammingDistance().get_raw_score("ciao", "ciro12"))


if __name__ == "__main__":
    main()
