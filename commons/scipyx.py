import numpy as np
import h5py
from scipy.optimize import *


def maximize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    return minimize(lambda x: -fun(x), x0, args=args, method=method, jac=jac, hess=hess,
                    hessp=hessp, bounds=bounds, constraints=constraints, tol=tol,
                    callback=callback, options=options)


def loadmat_hdf(file_name, mdict=None, appendmat=True, **kwargs):
    d = dict()
    with h5py.File(file_name, 'r') as f:
        for k in f.keys():
            data = f[k].value
            print(data.shape)
            data = np.swapaxes(data, 0, 1)
            print(data.shape)
            d[k] = data
    return d