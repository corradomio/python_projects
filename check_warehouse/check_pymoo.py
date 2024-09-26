import numpy as np
from pymoo.core.population import Population

from pymoox.math import prod_
from pymoox.operators.sampling.rnd import BinaryRandomSampling2D
from pymoox.operators.mutation.bitflip import BitflipMutation2D

class Problem:
    def __init__(self):
        self.shape_var = (4, 8)
        self.n_var = prod_(self.shape_var)


def main():
    p = Problem()
    # bs = BinaryRandomSampling2D(3)
    # bs.do(p, 10)
    #
    # bs = BinaryRandomSampling2D(3, axis=0)
    # bs.do(p, 10)
    #
    bs = BinaryRandomSampling2D(3, axis=1)
    pop: Population = bs.do(p, 10)

    mut = BitflipMutation2D(prob_var=0.1)
    mut.do(p, pop)

    mut = BitflipMutation2D(prob_var=0.1, axis=1)
    mut.do(p, pop)



    pass




if __name__ == "__main__":
    main()
