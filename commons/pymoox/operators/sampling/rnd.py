# from abc import ABC
# from random import shuffle
#
# import numpy as np
# from pymoo.core.sampling import Sampling
#
#
# class BinaryRandomSampling2D(Sampling, ABC):
#
#     def __init__(self, n_bools=0, axis=None):
#         super().__init__()
#         self.axis = axis
#         self.n_bools = n_bools
#
#     def _do(self, problem, n_samples, **kwargs):
#         n, m = problem.shape_var
#         nm = n*m
#
#         if (self.n_bools == 0 or self.n_bools >= nm) and self.axis is None:
#             val = np.random.random((n_samples, nm))
#             return (val < 0.5).astype(bool)
#
#         S = np.zeros((n_samples, n, m), dtype=bool)
#
#         if self.axis is None:
#             S = S.reshape((n_samples, -1))
#             n_bools = self.n_bools if 0 < self.n_bools <= nm else nm
#             indices = list(range(nm))
#             for i in range(n_samples):
#                 shuffle(indices)
#                 selected = indices[:n_bools]
#                 S[i, selected] = True
#             S = S.reshape((n_samples, n, m))
#         elif self.axis == 0:
#             n_bools = self.n_bools if 0 < self.n_bools <= n else n
#             indices = list(range(n))
#             for i in range(n_samples):
#                 for k in range(m):
#                     shuffle(indices)
#                     selected = indices[:n_bools]
#                     S[i, selected, k] = True
#         elif self.axis == 1:
#             n_bools = self.n_bools if 0 < self.n_bools <= m else m
#             indices = list(range(m))
#             for i in range(n_samples):
#                 for j in range(n):
#                     shuffle(indices)
#                     selected = indices[:n_bools]
#                     S[i, j, selected] = True
#         return S.reshape((n_samples, -1))
