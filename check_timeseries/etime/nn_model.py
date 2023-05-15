# 'sktime' protocols
#
#       cutoff
#       fh
#
#       model.fit(y, X=None, fh=None)
#       model.predict(fh=None, X=None)
#
#       fit_predict(y, X=None, fh=None):
#
#       update(y, X=None, update_params=True)
#       update_predict(y, cv=None, X=None, update_params=True, reset_forecaster=True)
#       update_predict_single(y=None, fh=None, X=None, update_params=True)
#
# In 'Time Series Forecasting in Python', the library predict MULTIPLE
# predictions in a single step
# In 'sktime' it is used a 'autoregressive approach: it is predictied just a single
# next timeslot, then this timeslot (and k-1 previous) is used to predict the next,
# etc
#
# To create a Deep Learning model we need to CREATE the MODEL and to COMPILE it
# using a LOSS function AND an OPTIMIZER.

from datetime import datetime
from typing import Union, Optional

import numpy as np
import pandas as pd
import sktime.forecasting.base as skf
from pandas import PeriodIndex
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from stdlib import import_from
import torch.nn as nn

from .lag import resolve_lag, LagTrainTransform, LagPredictTransform


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class DataWindow:
    def __init__(self, lag):
        self._lag = lag
        self.slots = resolve_lag(lag)

        self.y = None
        self.X = None

    def fit(self, y, X=None):
        self.y = y
        self.X = X
        return self

    def transform(self, y, X=None) -> np.ndarray:
        pass


class TSTorchModule(nn.Module):
    def __init__(self, model: list[str, dict]):
        super().__init__()

        self._model_config = model
        self._model = None

        self._compose_model()

    def _compose_model(self):
        layers = []
        for layer_config in self._model_config:
            layer = create_layer(layer_config)
        self._model = nn.Sequential(layers)


# ---------------------------------------------------------------------------
# DeppForecastRegressor
# ---------------------------------------------------------------------------
# model: how many layers, layer's types, layer's sizes
# optimizer: which optimizer to use
# loss: which loss function to use
#
# note: some 'predefined' models can have a 'name'
#

class DeepForecastRegressor(BaseForecaster):
    def __init__(self,
                 lag: Union[int, list, tuple, dict],
                 target: Optional[str] = None,

                 model: Optional[dict] = None,
                 optimizer: Optional[dict] = None,
                 loss: Optional[dict] = None,
                 **kwargs):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        # reorganize X to


        return self