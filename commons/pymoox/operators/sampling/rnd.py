from typing import Optional

import numpy as np
from random import shuffle, randrange, choice
from pymoo.core.sampling import Sampling


class BinaryRandomSampling2D(Sampling):
    """
    Create a population of matrices where in each matrix's column contains a single True
    """
    def __init__(self, D: Optional[np.ndarray]=None, wmax: int=-1):
        super().__init__()
        self.D = D
        self.wmax = wmax

    def _do(self, problem, n_samples, **kwargs):
        n, m = problem.shape_var
        wmax = self.wmax
        wmax = wmax if 0 < wmax <= n else n

        wall = list(range(n))

        T = np.zeros((n_samples, n, m), dtype=bool)
        for i in range(n_samples):
            shuffle(wall)
            nw = randrange(1, wmax)
            sel = wall[:nw]
            for k in range(m):
                j = choice(sel)
                T[i, j, k] = True

        if self.D is not None:
            self._show_best_solution(T)

        return T.reshape((n_samples, -1))
    # end

    def _show_best_solution(self, T):
        D = self.D
        n_samples, n, m = T.shape

        bestdi = float('inf')
        bestTi = None

        for i in range(n_samples):
            Ti = T[i]
            di = (D*Ti).sum()
            print(f"{i:4} {Ti.sum(axis=1)} {di:8.3f}")
            if di < bestdi:
                bestdi = di
                bestTi = Ti

        print(f"best {bestTi.sum(axis=1)} {bestdi:8.3f}")
    # end
# end
