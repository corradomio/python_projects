import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from common import Data
from pymoox.operators.crossover.pntx import TwoPointCrossover2D
from pymoox.operators.mutation.bitflip import BitflipMutation2D
from pymoox.operators.sampling.rnd import BinaryRandomSampling2D
from pymoox.problems.functional import FunctionalProblem


class NumOfWarehouses:
    def __init__(self, i):
        self.i = i
    def __call__(self, T):
        i = self.i
        n = T[:, i].sum()
        return 1 - n


class MaxWarehouses:
    def __init__(self, wmax):
        self.wmax = wmax
        pass
    def __call__(self, T):
        wmax = self.wmax
        n = np.sign(T.sum(1)).sum()
        return n - wmax


class MinDistance:
    def __init__(self, D: np.ndarray):
        self.D = D

    def __call__(self, T: np.ndarray):
        D = self.D
        md = (T*D).sum()
        return md


class MinNeighbourhood:
    def __init__(self, D: np.ndarray, dmin: float):
        self.D = D
        self.dmin = dmin
        self.F = (D<dmin)

    def __call__(self, T: np.ndarray):
        # F = self.F
        # tmin = (F*(1-T)).sum()

        F = self.F
        t = np.sign(T.sum(axis=1))
        nmin = t.dot(F*(1-T)).sum()
        return nmin



class LocationsServed(FunctionalProblem):

    def __init__(self, D: np.ndarray, dmin: float=0., wmax: int=-1):
        assert dmin >= 0
        # n: n of warehouses
        # m: n of locations
        n, m = D.shape
        wmax = wmax if 0 < wmax <= n else n
        super().__init__(
            objs = [
                MinDistance(D),
                MinNeighbourhood(D, dmin)
            ],
            constr_eq=[
                NumOfWarehouses(i)
                for i in range(m)
            ],
            constr_ieq=[
                MaxWarehouses(wmax)
            ],
            n_var=(n, m), vtype=bool,
            xl=False, xu=True
        )
        self.D: np.ndarray = D
        self.dmin: float = dmin
        self.wmax: int = wmax
    # end
# end


# class BinaryRandomSampling2D_(Sampling):
#     def __init__(self, D, wmax: int=-1):
#         super().__init__()
#         self.D = D
#         self.wmax = wmax
#
#     def _do(self, problem, n_samples, **kwargs):
#         D = self.D
#         n, m = problem.shape_var
#         wmax = self.wmax
#         wmax = wmax if 0 < wmax <= n else n
#
#         bestdi = 100000000
#         bestTi = None
#
#         wall = list(range(n))
#
#         T = np.zeros((n_samples, n, m), dtype=bool)
#         for i in range(n_samples):
#             shuffle(wall)
#             nw = randrange(1, wmax)
#             sel = wall[:nw]
#             for k in range(m):
#                 j = choice(sel)
#                 T[i, j, k] = True
#
#             Ti = T[i]
#             di = (D*Ti).sum()
#             print(f"{i:4} {Ti.sum(1)} {di:8.3f}")
#             if di < bestdi:
#                 bestdi = di
#                 bestTi = Ti
#         print(f"best {bestTi.sum(1)} {bestdi:8.3f}")
#         return T.reshape((n_samples, -1))


# class BitflipMutation2D_(Mutation):
#
#     def _do(self, problem, X, **kwargs):
#         n_samples, _ = X.shape
#         n, m = problem.shape_var
#         X = X.reshape((n_samples, n, m))
#         pro_var = self.prob_var.get()
#
#         for i in range(n_samples):
#             for k in range(m):
#                 if X[i, :, k].sum() == 1 and random() >= pro_var:
#                     continue
#                 # sel = np.nonzero(X[i].sum(axis=1))
#                 X[i, :, k] = False
#                 # j = choice(sel)
#                 j = randrange(n)
#                 X[i, j, k] = True
#         # end
#         X = X.reshape((n_samples, -1))
#         return X


# class TwoPointCrossover2D_(Crossover):
#
#     def __init__(self, n_points=2, **kwargs):
#         super().__init__(2, 2, **kwargs)
#         self.n_points = n_points
#
#     def _do(self, problem, X, **kwargs):
#         n_parents, n_matings, _ = X.shape
#         n, m = problem.shape_var
#         X = X.reshape((n_parents, n_matings, n, m))
#
#         # start point of crossover
#         r = np.vstack([np.random.permutation(m - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
#         r.sort(axis=1)
#         r = np.column_stack([r, np.full(n_matings, m)])
#
#         # the mask do to the crossover
#         M = np.full((n_matings, n, m), False)
#
#         # create for each individual the crossover range
#         for i in range(n_matings):
#
#             j = 0
#             while j < r.shape[1] - 1:
#                 a, b = r[i, j], r[i, j + 1]
#                 M[i, :, a:b] = True
#                 j += 2
#
#         Xp = crossover_mask(X, M)
#
#         Xp = Xp.reshape((n_parents, n_matings, -1))
#         return Xp


def result_reshape(X: np.ndarray, shape) -> np.ndarray:
    if len(X.shape) == 1:
        return X.reshape((1,) + shape)
    else:
        n = len(X)
        return X.reshape((n,) + shape)


def solve_locations_served(data, dmin: float=0., wmax: int = -1):
    D = data.distances(locations=True)

    problem = LocationsServed(D, dmin, wmax)

    algorithm = NSGA2(
        pop_size=200,
        # sampling=BinaryRandomSampling2D(wmax, axis=1),
        sampling=BinaryRandomSampling2D(D, wmax),
        crossover=TwoPointCrossover2D(),
        mutation=BitflipMutation2D(prob_var=0.1),
        eliminate_duplicates=True
    )

    res: Result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=get_termination("n_gen", 500),
        # seed=1,
        verbose=True
    )

    R: np.ndarray = result_reshape(res.X.astype(np.int8), (10, -1))
    # (n_sol, n, m)

    n_sol = len(R)
    for i in range(n_sol):
        # print("Best solution found: \nX = %s\nF = %s" % (R, res.F))
        # print(R[i].sum(axis=-2))
        print(R[i].sum(axis=-1), res.F[i])
    # print("... G = %s\n... H = %s" % (res.G, res.H))


def main():
    data: Data = Data().load("_10")
    T = data.clusters()
    D = data.distances()
    C = data.best_center()
    print(T.sum(axis=1), (D*T).sum())
    print(C)

    # data.plot(around=0)

    solve_locations_served(data, dmin=0.0, wmax=-1)
    pass



if __name__ == "__main__":
    main()
