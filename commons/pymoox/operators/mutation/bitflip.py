import numpy as np
from random import random, randrange
from pymoo.core.mutation import Mutation


class BitflipMutation2D(Mutation):

    def _do(self, problem, X: np.ndarray, **kwargs):
        n_samples, _ = X.shape
        n, m = problem.shape_var
        X = X.reshape((n_samples, n, m))
        pro_var = self.prob_var.get()

        for i in range(n_samples):
            for k in range(m):
                if X[i, :, k].sum() == 1 and random() >= pro_var:
                    continue
                # sel = np.nonzero(X[i].sum(axis=1))
                X[i, :, k] = False
                # j = choice(sel)
                j = randrange(n)
                X[i, j, k] = True
        # end
        X = X.reshape((n_samples, -1))
        return X
