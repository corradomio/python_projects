import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.split.base import BaseWindowSplitter
from sktime.split.slidingwindow import SlidingWindowSplitter as Sktime_SlidingWindowSplitter
from sktime.split.base._common import (
    DEFAULT_FH,
    DEFAULT_STEP_LENGTH,
    DEFAULT_WINDOW_LENGTH,
    FORECASTING_HORIZON_TYPES,
    SPLIT_GENERATOR_TYPE, ACCEPTED_Y_TYPES,
)
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    NON_FLOAT_WINDOW_LENGTH_TYPES,
)


class SlidingWindowSplitter(Sktime_SlidingWindowSplitter):
    """
    Force fh to be a ForecastingHorizon
    """
    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES | None = None,
        start_with_window: bool = True
    ):
        if isinstance(fh, int):
            fh = list(range(1, fh+1))
        if isinstance(fh, list):
            fh = ForecastingHorizon(fh)
        assert isinstance(fh, ForecastingHorizon)
        super().__init__(
            fh=fh,
            window_length=window_length,
            step_length=step_length,
            initial_window=initial_window,
            start_with_window=start_with_window,
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


class CountWindowSplitter(BaseWindowSplitter):
    """
    Dynamic step_length to create n_splits
    """
    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES | None = None,
        start_with_window: bool = True,
        n_splits: int = 5,
    ):
        if isinstance(fh, int):
            fh = list(range(1, fh+1))
        if isinstance(fh, list):
            fh = ForecastingHorizon(fh)
        assert isinstance(fh, ForecastingHorizon)
        assert n_splits >= 1
        super().__init__(
            fh=fh,
            window_length=window_length,
            step_length=1,
            initial_window=initial_window,
            start_with_window=start_with_window,
        )
        self.n_splits = n_splits

    def _split_windows(
            self,
            window_length: ACCEPTED_WINDOW_LENGTH_TYPES,
            y: pd.Index,
            fh: ForecastingHorizon,
    ) -> SPLIT_GENERATOR_TYPE:
        self._compute_step_length(y, window_length, fh)
        return self._split_windows_generic(
            window_length=window_length,
            y=y,
            fh=fh,
            expanding=False,
        )

    def get_n_splits(self, y: ACCEPTED_Y_TYPES | None = None):
        self._compute_step_length(y)
        return super().get_n_splits(y)

    def _compute_step_length(self, y, window_length=None,fh=None):
        window_length = self.window_length if window_length is None else window_length
        fh = self.fh if fh is None else fh
        rest_length = len(y) - (window_length + len(fh))
        step_length = rest_length // self.n_splits
        self.step_length = step_length if step_length > 1 else 1