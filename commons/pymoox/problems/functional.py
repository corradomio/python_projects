import pymoo.problems.functional as moopf
import numpy as np
from ..math import prod_


class FunctionalProblem(moopf.FunctionalProblem):
    def __init__(self,
                 n_var,
                 objs,
                 constr_ieq=[],
                 constr_eq=[],
                 func_pf=moopf.func_return_none,
                 func_ps=moopf.func_return_none,
                 **kwargs):
        self.shape_var = n_var
        self._reshape = (lambda x:x) if isinstance(n_var, int) else (lambda x:x.reshape(n_var))
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
        x = self._reshape(x)
        return super()._evaluate(x, out, *args, **kwargs)
