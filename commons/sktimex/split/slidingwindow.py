import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.split.slidingwindow import SlidingWindowSplitter as Sktime_SlidingWindowSplitter
from sktime.split.base._common import (
    DEFAULT_FH,
    DEFAULT_STEP_LENGTH,
    DEFAULT_WINDOW_LENGTH,
    FORECASTING_HORIZON_TYPES,
    SPLIT_GENERATOR_TYPE,
)
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    NON_FLOAT_WINDOW_LENGTH_TYPES,
)


class SlidingWindowSplitter(Sktime_SlidingWindowSplitter):
    """
    Extends 'sktime.split.SlidingWindowSplitter' adding support for
    'window_length' and 'step_length' specified as ratio values in range (0,1)
    """
    def __init__(
            self,
            fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
            window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
            step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
            initial_window: ACCEPTED_WINDOW_LENGTH_TYPES | None = None,
            start_with_window: bool = True,
    ):
        super().__init__(
            fh=list(range(1,fh+1)) if isinstance(fh, int) else fh,
            window_length=window_length,
            step_length=step_length,
            initial_window=initial_window,
            start_with_window=start_with_window
        )
        pass
    # end

    def _split_windows(
            self,
            window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
            y: pd.Index,
            fh: ForecastingHorizon,
    ) -> SPLIT_GENERATOR_TYPE:
        return self._split_windows_generic(
            window_length=window_length,
            y=y,
            fh=fh,
            expanding=False,
        )

