from scipy.optimize import *


def maximize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    return minimize(lambda x: -fun(x), x0, args=args, method=method, jac=jac, hess=hess,
                    hessp=hessp, bounds=bounds, constraints=constraints, tol=tol,
                    callback=callback, options=options)
