from joblibx import Parallel, delayed
from pprint import pprint


def random(i):
    # rnd = Random(i)
    # return round(rnd.random(), 4)
    return i


def main():
    # rnd = Random()
    # r = []
    # for i in range(10):
    #     r.append(round(rnd.random(), 4))

    r = Parallel(n_jobs=(10, 2))(delayed(random)(i) for i in range(100000))

    pprint(r)


if __name__ == "__main__":
    main()
