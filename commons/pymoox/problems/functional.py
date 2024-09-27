import pymoo.problems.functional as moopf
import numpy as np
from ..math import prod_


class FunctionalProblem(moopf.FunctionalProblem):
    """
    Extends pymoo::FunctionalProblem supporting 2D variables.
    The parameter 'n_var' must be a tuple of integers, saved
    in 'shape_var', and 'n_vars' is the product of all integers

    The single solution X is passed to objective functions and
    constraints as matrix (or tensor)

    The class offer 3 'evaluation' functions to speedup objective
    and contraints evaluations:

        - _evaluate_objs
        - _evaluate_constr_ieq
        - _evaluate_constr_eq

    Each function must return an array.
    """
    def __init__(self,
                 n_var,
                 objs,              # [f1(x), ...]
                 constr_ieq=[],     # [g1(x) <= 0, ...]
                 constr_eq=[],      # [h1(x) == 0, ...]
                 func_pf=moopf.func_return_none,
                 func_ps=moopf.func_return_none,
                 **kwargs):
        assert isinstance(n_var, tuple)
        self.shape_var = n_var
        super().__init__(
            n_var=prod_(n_var),
            objs=objs,
            constr_ieq=constr_ieq,
            constr_eq=constr_eq,
            func_pf=func_pf,
            func_ps=func_ps,
            **kwargs
        )

    def _evaluate(self, x: np.ndarray, out: dict[str, np.ndarray], *args, **kwargs):
        x = x.reshape(self.shape_var)

        out["F"] = self._evaluate_objs(x)
        out["G"] = self._evaluate_constr_ieq(x)
        out["H"] = self._evaluate_constr_eq(x)
        # return super()._evaluate(x, out, *args, **kwargs)

    def _evaluate_objs(self, x):
        return np.array([obj(x) for obj in self.objs])

    def _evaluate_constr_ieq(self, x):
        return np.array([constr(x) for constr in self.constr_ieq])

    def _evaluate_constr_eq(self, x):
        return np.array([constr(x) for constr in self.constr_eq])
# end
