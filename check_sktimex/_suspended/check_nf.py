import logging
import matplotlib.pyplot as plt
import pandas as pd
from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import *
from neuralforecast.utils import AirPassengersDF

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# Split data and declare panel dataset
Y_df = AirPassengersDF
Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31'] # 12 test

# Fit and predict with NBEATS and NHITS models
horizon = len(Y_test_df)
# models = [NBEATS(input_size=2 * horizon, h=horizon, max_steps=100, enable_progress_bar=False),
#           NHITS(input_size=2 * horizon, h=horizon, max_steps=100, enable_progress_bar=False)]
models = [
    # Autoformer(input_size=2 * horizon, h=horizon, max_steps=100, enable_progress_bar=True)
    # MLP(input_size=2 * horizon, h=horizon, max_steps=100, enable_progress_bar=True)
    NBEATS(input_size=14, h=1, max_steps=100, enable_progress_bar=True, stack_types=[])
]

nf = NeuralForecast(models=models, freq='ME')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict()

# Plot predictions
plot_series(Y_train_df, Y_hat_df)
plt.show()
