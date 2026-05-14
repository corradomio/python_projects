import sktime.forecasting.theta as skf


class ThetaForecaster(skf.ThetaForecaster):
    def __init__(
        self,
        initial_level=None,
        deseasonalize=True,
        sp=1,
        deseasonalize_model="multiplicative",
    ):
        super().__init__(
            initial_level=initial_level,
            deseasonalize=bool(deseasonalize),
            sp=int(sp),
            deseasonalize_model=str(deseasonalize_model)
        )
