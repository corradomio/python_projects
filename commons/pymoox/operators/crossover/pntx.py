# import numpy as np
#
# from pymoo.core.crossover import Crossover
# from pymoo.util.misc import crossover_mask
#
#
# class TwoPointCrossover2D(Crossover):
#
#     def __init__(self, axis=None, n_points=2, **kwargs):
#         super().__init__(2, 2, **kwargs)
#         self.axis = axis
#         self.n_points = n_points
#
#     def _do(self, problem, X, **kwargs):
#         if self.axis is None:
#             return self._do_plain(problem, X, **kwargs)
#         elif self.axis == 0:
#             return self._do_axis_0(problem, X, **kwargs)
#         elif self.axis == 1:
#             return self._do_axis_1(problem, X, **kwargs)
#
#     def _do_plain(self, problem, X, **kwargs):
#         n, m = problem.shape_var
#
#         # get the X of parents and count the matings
#         n_parents, n_matings, n_var = X.shape
#
#         # start point of crossover
#         r = np.row_stack([np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
#         r.sort(axis=1)
#         r = np.column_stack([r, np.full(n_matings, n_var)])
#
#         # the mask do to the crossover
#         M = np.full((n_matings, n_var), False)
#
#         # create for each individual the crossover range
#         for i in range(n_matings):
#
#             j = 0
#             while j < r.shape[1] - 1:
#                 a, b = r[i, j], r[i, j + 1]
#                 M[i, a:b] = True
#                 j += 2
#
#         Xp = crossover_mask(X, M)
#
#         return Xp
#
#     def _do_axis_0(self, problem, X, **kwargs):
#         return self._do_plain(problem, X, **kwargs)
#
#     def _do_axis_1(self, problem, X, **kwargs):
#         return self._do_plain(problem, X, **kwargs)
