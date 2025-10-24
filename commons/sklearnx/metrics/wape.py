from typing import Union
import numpy as np
import pandas as pd


#
# https://www.baeldung.com/cs/mape-vs-wape-vs-wmape
# A: actuale   y_true
# F: forecast  y_pred
#
#   MAPE = 1/n SUM(t, ABS((At - Ft)/At)
#
#   WAPE = SUM(t, ABS(At - Ft))/SUM(t, ABS(At))

def weighted_absolute_percentage_error(
        y_true: Union[pd.DataFrame, pd.Series],
        y_pred: Union[pd.DataFrame, pd.Series]
):
    # normalize
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.to_numpy()

    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape, (
        f"y_true.shape={y_true.shape} != y_pred.shape={y_pred.shape}"
    )

    if len(y_true.shape) == 2:
        y_true = y_true.reshape(-1)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(-1)

    total = np.abs(y_true).sum()
    diff = np.abs(y_true - y_pred).sum()
    return diff/total if total > 0 else diff
# end

#
# From Sid library
#
WAPE_INFINITY = 10000.

def weighted_absolute_percentage_error_sid(
        y_true,
        y_pred
):
    if len(y_true) == 0:
        return WAPE_INFINITY

    y_true = y_true.values.reshape(-1)
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values.reshape(-1)
    elif isinstance(y_pred, np.ndarray):
        y_pred = y_pred
    elif isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    else:
        y_pred = y_pred

    num = abs(y_true - y_pred).sum()
    den = abs(y_true).sum()
    if den == 0:
        return WAPE_INFINITY
    else:
        return num / den
# end
