# import numpy as np
# from pymoo.core.mutation import Mutation
# from pymoo.core.variable import get
#
#
# class BitflipMutation2D(Mutation):
#     def __init__(self, axis=None, prob=1.0, prob_var=None, name=None, vtype=None, repair=None):
#         super().__init__(prob=prob, prob_var=prob_var, name=name, vtype=vtype, repair=repair)
#         self.axis = axis
#
#     def _do(self, problem, X, **kwargs):
#         prob_var = self.get_prob_var(problem, size=(len(X), 1))
#         Xp = np.copy(X)
#         flip = np.random.random(X.shape) < prob_var
#         # if flip.sum() > 0:
#         #     print("flip:", flip.sum()/len(X))
#         Xp[flip] = ~X[flip]
#         return Xp
#
#     def get_prob_var(self, problem, **kwargs):
#         if self.prob_var is None:
#             n, m = problem.shape_var
#             if self.axis is None:
#                 prob_var = min(0.5, 1 / problem.n_var)
#             elif self.axis == 0:
#                 prob_var = min(0.5, 1 / n)
#             elif self.axis == 1:
#                 prob_var = min(0.5, 1 / m)
#             else:
#                 prob_var = min(0.5, 1 / problem.n_var)
#         else:
#             prob_var = self.prob_var
#         return get(prob_var, **kwargs)
