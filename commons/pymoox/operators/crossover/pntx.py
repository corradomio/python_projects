import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


class TwoPointCrossover2D(Crossover):

    def __init__(self, n_points=2, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, _ = X.shape
        n, m = problem.shape_var
        X = X.reshape((n_parents, n_matings, n, m))

        # start point of crossover
        r = np.vstack([np.random.permutation(m - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, m)])

        # the mask do to the crossover
        M = np.full((n_matings, n, m), False)

        # create for each individual the crossover range
        for i in range(n_matings):

            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[i, :, a:b] = True
                j += 2

        Xp = crossover_mask(X, M)

        Xp = Xp.reshape((n_parents, n_matings, -1))
        return Xp
