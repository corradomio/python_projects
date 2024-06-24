import sktime.forecasting.base as skt


class ForecastingHorizon(skt.ForecastingHorizon):
    """
    Extends sktime ForecastingHorizon in the following way:
    """

    def __init__(self, values=None, is_relative=None, freq=None):
        """
        Enhanced version of Sktime ForecastingHorizon

        :param values: if it is an integer, it is converted in the list [1,..value]

        :param is_relative:
        :param freq:
        """
        if isinstance(values, int):
            super().__init__(list(range(1, values+1)), freq=freq, is_relative=True)
        else:
            super().__init__(values, is_relative=is_relative, freq=freq)

    @property
    def freq(self) -> str:
        # return super().freq
        return super(ForecastingHorizon, self.__class__).freq.fget(self)

    @freq.setter
    def freq(self, value) -> None:
        if value == 'M':
            value = 'MS'
        super(ForecastingHorizon, self.__class__).freq.fset(self, value)

