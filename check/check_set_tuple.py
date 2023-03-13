from random import randrange
from time import time


def main():
    m = 100000
    n = 10000
    S = set(range(n))

    start = time()
    for i in range(m):
        e = randrange(n)
        if e in S:
            pass
    # end
    delta = time() - start
    print(delta, "s")


if __name__ == "__main__":
    main()
