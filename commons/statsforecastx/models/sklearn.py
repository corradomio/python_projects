import statsforecast.models as stfm

from stdlib.qname import create_from
from .base import _BaseStatsForecastForecaster


class SklearnModel(_BaseStatsForecastForecaster):
    def __init__(
            self,
            model,
            # prediction_intervals: Optional[ConformalIntervals] = None,
            # alias: Optional[str] = None,
    ):
        super().__init__(stfm.SklearnModel, locals())
        return

    def _validate_kwargs(self, stf_kwargs: dict, y, X) -> dict:
        model_config = stf_kwargs["model"]
        stf_kwargs["model"] = create_from(model_config)
        return stf_kwargs
    # end
