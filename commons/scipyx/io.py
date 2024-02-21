import numpy as np
import h5py


def loadmat_hdf(file_name, mdict=None, appendmat=True, **kwargs):
    d = dict()
    with h5py.File(file_name, 'r') as f:
        for k in f.keys():
            data = f[k]
            data = data[:]
            print(data.shape)
            data = np.swapaxes(data, 0, 1)
            print(data.shape)
            d[k] = data
    return d