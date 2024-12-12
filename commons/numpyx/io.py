from typing import Optional

from numpy import asarray

import csvx


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def load_data(fname: str, ycol=-1, dtype=None, skiprows=0, na: Optional[str] = None):
    data, _ = csvx.load(fname, dtype=dtype, skiprows=skiprows, na=na)

    data = asarray(data)
    nr, nc = data.shape

    if ycol == 0:
        X = data[:, 1:]
        y = data[:, 0]
    elif ycol == -1 or ycol == (nc - 1):
        X = data[:, 0:-1]
        y = data[:, -1]
    else:
        X = data[:, list(range(0, ycol)) + list(range(ycol + 1, nc))]
        y = data[:, ycol]

    if ycol == 0:
        dtypes = list(set(dtype[1:]))
    elif ycol == -1:
        dtypes = list(set(dtype[0:-1]))
    else:
        dtypes = list(set(dtype[0:ycol] + dtype[ycol + 1:]))
    if len(dtypes) == 1 and dtypes[0] in ["enum", "ienum", enumerate]:
        X = X.astype(int)

    if dtype[ycol] in ["enum", "ienum", str, int, enumerate]:
        y = y.astype(int)

    return X, y
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
