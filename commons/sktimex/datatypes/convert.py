import pandas as pd
import sktime.datatypes._convert as sktdtc
from sktime.forecasting.base import ForecastingHorizon


def convert_to(
    y_pred,
    to_type: str,
    as_scitype: str = None,
    store=None,
    store_behaviour: str = None,
    return_to_mtype: bool = False,
    y_cutoff=None
):
    """
    This function resolve a problem with the original one that, when it convert
    something in a series/dataframe, doesn't assign the correct index.
    Here, the correct index in extracted from X, if available, or, it is created
    using cutoff AND fh in relative way
    """
    y_out = sktdtc.convert_to(
        y_pred,
        to_type=to_type,
        as_scitype=as_scitype,
        store=store,
        store_behaviour=store_behaviour,
        return_to_mtype=return_to_mtype,
    )

    n_out = len(y_pred)
    index = ForecastingHorizon(list(range(1, n_out + 1))).to_absolute(y_cutoff).to_pandas()

    if isinstance(y_out, pd.Series):
        y_out = y_out.set_axis(index)
    elif isinstance(y_out, pd.DataFrame):
        y_out.set_index(index, inplace=True)

    return y_out
# end
