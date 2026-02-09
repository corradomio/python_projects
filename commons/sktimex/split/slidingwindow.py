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
    def __init__(
            self,
            fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
            window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
            step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
            initial_window: ACCEPTED_WINDOW_LENGTH_TYPES | None = None,
            start_with_window: bool = True,
    ):
        super().__init__(
            fh=fh,
            window_length=window_length if window_length >= 1 else int(100*window_length),
            step_length=step_length if step_length >= 1 else int(100*step_length),
            initial_window=initial_window,
            start_with_window=start_with_window
        )
        self._window_length_override = window_length
        self._step_length_override = step_length
        self._is_ratio = (0 < window_length < 1) or (0 < step_length < 1)
        pass
    # end

    def _split_windows(
            self,
            window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
            y: pd.Index,
            fh: ForecastingHorizon,
    ) -> SPLIT_GENERATOR_TYPE:
        if self._is_ratio:
            window_length = int(len(y)*self._window_length_override)

        return self._split_windows_generic(
            window_length=window_length,
            y=y,
            fh=fh,
            expanding=False,
        )

