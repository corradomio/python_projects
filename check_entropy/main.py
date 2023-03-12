#
# 01/10/2019 Corrado Mio (Local)
#
from info_theory import *


def main():
    n = 30
    p1 = [1]+[0]*(n-1)
    p2 = [1/n]*n
    print("---")
    print(entropy(p1))
    print(entropy(p2))
    print(-log2(1/n))
    print(log2(n))
    print("---")
    print(euclidean_norm(p1))
    print(euclidean_norm(p2))
    print(1/n*sqrt(n))
    print("---")


if __name__ == "__main__":
    main()
