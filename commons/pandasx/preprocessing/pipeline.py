from pandas import DataFrame
from .base import BaseEncoder


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:

    def __init__(self, steps: list[BaseEncoder]):
        self._steps: list[BaseEncoder] = steps

    def fit(self, X: DataFrame) -> "Pipeline":
        for step in self._steps:
            X = step.fit_transform(X)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        for step in self._steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X: DataFrame) -> DataFrame:
        for step in self._steps:
            X = step.fit_transform(X)
        return X
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
