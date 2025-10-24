import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from stdlib.tprint import tprint

from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test
Y_test_df.drop(columns=['y'], inplace=True)

tprint("Create model")
model = MLP(h=12, input_size=24,
            loss=DistributionLoss(distribution='Normal', level=[80, 90]),
            scaler_type='robust',
            learning_rate=1e-3,
            max_steps=200,
            val_check_steps=300,
            early_stop_patience_steps=2)

tprint("Create wrapper")
fcst = NeuralForecast(
    models=[model],
    freq='M'
)

tprint("fit")
fcst.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
tprint("predict")
forecasts = fcst.predict(futr_df=Y_test_df)
tprint("done")

