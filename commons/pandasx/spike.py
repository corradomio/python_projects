from typing import Optional, Union

import numpy as np
import pandas as pd


def is_spike(df: Union[pd.Series, pd.DataFrame], *, target: Optional[str] = None,
             outlier_std: float = 3.0,
             outliers: float = 0.1,
             method='mean',
             detrend=False) -> bool:
    """
    Check if the selected series/column is a 'spike', that is, it is almost constant
    except for some outliers
    The data is considered a 'spike' if 'tau*min_diff < max_diff'

    :param X: data to analyze
    :param outlier_std: factor used to multiply the standard deviation to compute minimum and maximum
        values.
    :param method: which statistical function to use (mean, median, min)
    :param outliers: maximum number of outliers respect the number of elements.
        Can be specified in relative way (in range (0, 1)) or in absolute way (>= 1)
    :param detrend: if to remove the 'trend', using a simple linear regression
    :return: True if the data seems to be a spike
    """
    assert isinstance(df, (pd.Series, pd.DataFrame))

    if target is not None:
        y = df[target].values
    else:
        y = df.values

    if outliers < 1:
        outliers = max(1, int(outliers*len(y)))

    if detrend:
        x = np.arange(len(y))
        m, b = np.polyfit(x, y, 1)
        y = y - m*x

    if method is None:
        y_mean = y.mean()
    elif method == 'mean':
        y_mean = y.mean()
    elif method == 'median':
        y_mean = y.median()
    elif method == 'min':
        y_mean = y.min()
    else:
        raise ValueError(f'Unsupported method {method}')
    # end

    y_std = y.std()
    y_min = y_mean - outlier_std * y_std
    y_max = y_mean + outlier_std * y_std

    n_outliers = len(y[y < y_min]) + len(y[y > y_max])

    return 0 < n_outliers < outliers
# end
