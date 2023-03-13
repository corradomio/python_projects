#
# Check  Weighted Random Set
#
import numpy as np
from mathx import comb, sumcomb
from random import Random
from randomx import WeightedRandomSets, SetsStatistics


def test_pitems():
    n = 10
    m = 10000

    wrs = WeightedRandomSets(n)
    # wrs = WeightedRandomSets(n, k=None, pitems=[.1, .2, .3, .4, .5, .6, .7, .6, .5, .4])
    # wrs = WeightedRandomSets(n, k=[5], pitems=[.1, .2, .3, .4, .5, .6, .7, .6, .5, .4])
    # wrs = WeightedRandomSets(n, k=5, pitems=[.1, .2, .3, .4, .5, .6, .7, .6, .5, .4])
    cnt = SetsStatistics().set(wrs)
    for i in range(m):
        S = wrs.next()
        cnt.add(S)
    # end
    cnt.report(prec=1)


def test_plevels():
    print("== test3 ==")

    n = 10
    m = 100000

    for i in range(n+1):
        # wrs = WeightedRandomSets(n, k=i)
        # wrs = WeightedRandomSets(n, k=[i])
        wrs = WeightedRandomSets(n, k=[i], plevels=[.1, .2, .3, .4, .5, .6, .7, .6, .5, .4, .3])
        cnt = SetsStatistics().set(wrs)
        for i in range(m):
            S = wrs.next()
            cnt.add(S)
        # end
        cnt.report(prec=3)
        print()


def main():
    # test_pitems()
    test_plevels()
    pass


if __name__ == "__main__":
    main()
