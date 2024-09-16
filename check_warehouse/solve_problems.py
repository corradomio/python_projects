from random import random

import numpy as np

from stdlib.jsonx import load


class Data:

    def __init__(self, wlist, llist, D, C):
        self.wlist = wlist
        self.llist = llist
        self.D = D
        self.C = C


def load_data() -> Data:
    wld = load("warehouses_locations.json")
    wlist = sorted(wld["warehouses"].keys())
    llist = sorted(wld["locations"].keys())
    dmap = wld["distances"]

    n = len(wlist)
    m = len(llist)

    D = np.zeros((n, m), dtype=float)
    C = np.zeros((n, m), dtype=bool)
    PROB = 0.75

    wdict = {wlist[i]: i for i in range(n)}
    ldict = {llist[i]: i for i in range(m)}

    for w in dmap:
        lmap = dmap[w]
        for l in lmap:
            if l not in ldict: continue
            i = wdict[w]
            j = ldict[l]
            r = random()

            D[i,j] = lmap[l]
            C[i,j] = r < PROB
        # end
    # end
    return Data(wlist, llist, D, C)


def main():
    data: Data = load_data()
    pass



if __name__ == "__main__":
    main()