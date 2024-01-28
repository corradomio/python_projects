from random import random
import numpy as np
from numpy import ndarray
from iset import *
from sfun import *


def gen_mobius(n: int) -> MobiusTransform:
    N = isetn(n)
    p = N+1
    mt = np.zeros(p, dtype=float)
    for S in ipowersetn(n, empty=False):
        card = icard(S)
        mt[S] = 1/card*random()
    # end
    return MobiusTransform.from_data(mt)


def main():
    MT = gen_mobius(10)
    SF = MT.set_function()
    N = SF.N
    MF = SF.mobius_transform()
    err = np.abs(MT.data - MF.data).sum()
    print(N, SF[N], SF.max(), err)

    for S in ilexpowerset(N, empty=True, full=True, k=(2,3)):
        print(ilist(S))

    pass


if __name__ == "__main__":
    main()
