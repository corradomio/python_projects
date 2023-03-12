from mathx import *
from randomx import *
from itertoolsx import *


def main():
    for i in range(100+1):
        f = ifactorize(i)
        c = icompose(f)
        if c == i:
            print(i, f)
        else:
            print(i, f, c)

    # print(comb(100, 50))


if __name__ == "__main__":
    main()
