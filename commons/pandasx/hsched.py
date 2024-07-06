from typing import Union, Optional, cast
import statsmodels.api as sm

import numpy as np
import pandas as pd


def is_heteroschedastic(
    df: Union[pd.Series, pd.DataFrame],
    *,
    target: Optional[str] = None,
    p_value: float = 0.05,
    test='fm'
) -> bool:
    """
    Check if the data is 'heteroschedastic', that is, mean and variance changes in the time.
    It is used the 'Breusch-Pagan test', available in 'statsmodels':

        1) it is computed a linear model on 'y'
        2) it is computed the residual 'res = y - m x'
        3) it is computed a linear model on 'res ~ b_err + m_err x'

    now, it is used a Chi quadro test to check if 'm_err' is zero

    reference:
        https://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test
        https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_breuschpagan.html

    :param df:  data to analyze
    :param target: target column, if it is passed a dataframe
    :param p_value: p-value used as threshold
    :param test: which test to use. For limited number of values, it is better to use 'F-statistics'
            ('fm'), otherwise the 'LM test' ('lm')
    :return:
    """
    assert isinstance(df, (pd.Series, pd.DataFrame))

    if target is not None:
        y = df[target].values
    else:
        y = df.values

    n = len(y)
    x = np.ones((n, 2))
    x[:, 0] = np.arange(n)

    fit = sm.OLS(y, x).fit()
    het_breuschpagan = cast(tuple[float], sm.stats.het_breuschpagan(fit.resid, fit.model.exog))
    # het_breuschpagan = (lm, lm_pvalue, fvalue, f_pvalue)
    #   lm:         lagrange multiplier statistic
    #   lm_pvalue
    #   fvalue:     f-statistic of the hypothesis that the error variance does not depend on x
    #   f_pvalue
    #
    # Note: In the general description of LM test, Greene mentions that this test exaggerates
    #       the significance of results in small or moderately large samples.
    #       In this case the F-statistic is preferable.
    #       (from statsmodels documentation)

    return (het_breuschpagan[3] < p_value) if test == 'fm' else (het_breuschpagan[1] < p_value)
# end
