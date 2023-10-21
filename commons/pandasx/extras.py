from math import isnan
from typing import Union

import pandas as pd
import numpy as np
from math import sqrt


# ---------------------------------------------------------------------------
# cumulant, lift, prob
# ---------------------------------------------------------------------------

def _lift_cumulant(df: pd.DataFrame, select: list) -> tuple:
    def float_(x):
        x = float(x)
        # return 0. if isnan(x) else x
        return x

    if 'count' not in df:
        df['count'] = 1.

    n = len(select)

    total = df['count'].count()

    # group count
    gcount = df[select + ['count']].groupby(select).count() / total

    # single count
    scount = dict()
    for c in select:
        scount[c] = df[[c] + ['count']].groupby([c]).count() / total

    index = gcount.index
    cvalues = []
    lvalues = []
    for keys in index.values:
        gvalue = float_(gcount.loc[keys])
        sproduct = 1.

        for i in range(n):
            c = select[i]
            k = keys[i]
            svalue = float_(scount[c].loc[k])
            if isnan(svalue): svalue = 1.
            sproduct *= svalue

        cvalue = gvalue - sproduct
        cvalues.append(cvalue)

        lvalue = gvalue / sproduct if sproduct != 0. else 0.
        lvalues.append(lvalue)
    # end
    return index, cvalues, lvalues
# end


def cumulant(df: pd.DataFrame, select: list) -> pd.DataFrame:
    """
            cumulant(f1,..fk) = prob(f1,..fk) - (prob(f1)*...*prob(fk))

    :param df:
    :param select:
    :return:
    """
    index, cvalues, lvalues = _lift_cumulant(df, select)
    return pd.DataFrame(data={"cumulant": pd.Series(cvalues, index=index, name="cumulant")})
# end


def lift(df: pd.DataFrame, select: list) -> pd.DataFrame:
    """
                             prob(f1,...fk)
        lift(f1,...fk) = -----------------------
                          prob(f1)*...*prob(fk)

    :param df:
    :param select:
    :return:
    """
    index, cvalues, lvalues = _lift_cumulant(df, select)
    return pd.DataFrame(data={"lift": pd.Series(lvalues, index=index, name="lift")})
# end


def prob(df: pd.DataFrame, select: list) -> pd.Series:
    if 'count' not in df:
        df['count'] = 1.
    total = df['count'].count()
    gcount = df[select + ['count']].groupby(select).count() / total

    return gcount
# end


# ---------------------------------------------------------------------------
# partition_lengths
# partitions_split
# ---------------------------------------------------------------------------

def partition_lengths(n: int, quota: Union[int, list[int]]) -> list[int]:
    if isinstance(quota, int):
        quota = [1] * quota
    k = len(quota)
    tot = sum(quota)
    lengths = []
    for i in range(k - 1):
        l = int(n * quota[i] / tot + 0.6)
        lengths.append(l)
    lengths.append(n - sum(lengths))
    return lengths
# end


def partitions_split(*data_list: list[pd.DataFrame], partitions: Union[int, list[int]] = 1, index=None, random=False) \
        -> list[Union[pd.DataFrame, pd.Series]]:
    parts_list = []
    for data in data_list:
        parts = _partition_split(data, partitions=partitions, index=index, random=random)
        parts_list.append(parts)
    # end
    parts_list = list(zip(*parts_list))
    return parts_list
# end


def _partition_split(data: pd.DataFrame, partitions: Union[int, list[int]], index, random) -> list[pd.DataFrame]:
    n = len(data)
    indices = list(range(n))
    plengths = partition_lengths(n, partitions)
    pn = len(plengths)
    s = 0
    parts = []
    for i in range(pn):
        pl = plengths[i]
        if index is None:
            part = data.iloc[s:s + pl]
        else:
            part_index = index[s:s + pl]
            part = data.loc[part_index]
        # end
        parts.append(part)
        s += pl
    # end
    return parts
# end


# ---------------------------------------------------------------------------
# classification_quality
# ---------------------------------------------------------------------------

def classification_quality(pred_proba: Union[pd.DataFrame, pd.Series], target=None) -> pd.DataFrame:
    """
    Compute the classification quality (a number between 0 and 1) based on
    the euclidean distance, then assign an index (an integer in range [0, n))
    in such way that the best classification quality has index 0 and the worst
    index (n-1).

    :param pred_proba: the output of 'ml.pred_proba()'
    :return: an array (n, 2) where the first column contains the classification
        quality and the second columnt the quality index
    """
    assert isinstance(pred_proba, (pd.DataFrame, pd.Series))
    if target is None:
        target = 'pred_qual'

    n, c = pred_proba.shape
    t = sqrt(c) / c
    # create the result data structure
    cq = pd.DataFrame({}, index=pred_proba.index)
    # classification quality
    cq[target] = (np.linalg.norm(pred_proba.values, axis=1) - t) / (1 - t)
    # assign the original prediction indices
    # cq['origin'] = range(n)
    # order the classification qualities in desc order
    cq.sort_values(by=[target], ascending=False, inplace=True)
    # assign the quality index order
    cq['rank'] = range(n)
    # back to the original order
    cq = cq.loc[pred_proba.index]
    # remove the extra column
    # cq = cq[:, 0:2]
    # done
    return cq
# end
