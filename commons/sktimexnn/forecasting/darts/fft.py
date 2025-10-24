from typing import Optional

import darts.models.forecasting.fft as dm

from .base import _BaseDartsForecaster


class _FFT(dm.FFT):
    def __init__(
            self,
            nr_freqs_to_keep: Optional[int] = 10,
            required_matches: Optional[set] = None,
            trend: Optional[str] = None,
            trend_poly_degree: int = 3,
    ):
        super().__init__(
            nr_freqs_to_keep=nr_freqs_to_keep,
            required_matches=required_matches,
            trend=trend,
            trend_poly_degree=trend_poly_degree
        )

    def predict(
            self,
            n: int,
            series=None,
            num_samples: int = 1,
            verbose: Optional[bool] = None,
            show_warnings: bool = True,
            random_state: Optional[int] = None,
    ) -> "TimeSeries":
        return super().predict(
            n=n,
            num_samples=num_samples,
            verbose=verbose,
            show_warnings=show_warnings,
            random_state=random_state
        )


class FFT(_BaseDartsForecaster):

    _tags = {
        "capability:exogenous": False,
        "capability:future-exogenous": False
    }

    def __init__(
            self,
            nr_freqs_to_keep: Optional[int] = 10,
            required_matches: Optional[set] = None,
            trend: Optional[str] = None,
            trend_poly_degree: int = 3,
            # --
            scaler=None,
            # --
    ):
        super().__init__(_FFT, locals())
