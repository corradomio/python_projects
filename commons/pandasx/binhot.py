from pandas import DataFrame
from typing import Union
from .base import as_list, safe_sorted


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def bitsof(k, n):
    t = k
    b = [0] * n
    i = 0
    while t > 0:
        if t % 2 == 1:
            b[i] = 1
        t >>= 1
        i += 1
    return b


def nbitsof(n):
    b = 1
    m = 2
    while m < n:
        b += 1
        m += m
    return b


# ---------------------------------------------------------------------------
# binhot_encode
# ---------------------------------------------------------------------------

def binhot_encode(df: DataFrame, columns: Union[str, list[str]]) -> DataFrame:

    assert isinstance(df, DataFrame)
    columns = as_list(columns, 'columns')

    for col in columns:
        # check if the column contains ONLY 2 values
        uv = safe_sorted(df[col].unique())
        nv = len(uv)
        if nv <= 2:
            vmap = {uv[i]: i for i in range(nv)}
            df[col] = df[col].replace(vmap)
            continue

        nb = nbitsof(nv)

        # generate the columns
        for b in range(nb):
            bcol = f"{col}_{b}"
            df[bcol] = 0

        for i, v in enumerate(uv):
            vbits = bitsof(i, nb)
            for b in range(nb):
                bcol = f"{col}_{b}"
                if vbits[b] == 1:
                    df.loc[df[col] == v, bcol] = 1

        # remove column & clone to reduce fragmentation
        df = df[df.columns.difference([col])].copy()
        pass
    return df
# end
