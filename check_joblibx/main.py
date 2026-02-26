from joblibx import Parallel, delayed
from pprint import pprint
from stdlib.tprint import tprint


def random(i):
    # rnd = Random(i)
    # return round(rnd.random(), 4)
    return i


def main():
    # rnd = Random()
    # r = []
    # for i in range(10):
    #     r.append(round(rnd.random(), 4))

    tprint(f"parallel")

    r = Parallel(n_jobs=(10, 2))(delayed(random)(i) for i in range(100000))

    tprint(f"len={len(r)}")

    for i in range(len(r)):
        if i != r[i]:
            print(i)

    tprint("done")


if __name__ == "__main__":
    main()
