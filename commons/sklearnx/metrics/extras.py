from typing import Union
import pandas as pd
import numpy as np


#
# https://www.baeldung.com/cs/mape-vs-wape-vs-wmape
# A: actuale   y_true
# F: forecast  y_pred
#
#   MAPE = 1/n SUM(t, ABS((At - Ft)/At)
#
#   WAPE = SUM(t, ABS(At - Ft))/SUM(t, ABS(At))
#
def weighted_absolute_percentage_error(y_true: Union[pd.DataFrame, pd.Series], y_pred: Union[pd.DataFrame, pd.Series]):
    # normalize
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.to_numpy()
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(-1)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(-1)

    total = np.abs(y_true).sum()
    diff = np.abs(y_true - y_pred).sum()
    return diff/total if total > 0 else diff
# end
