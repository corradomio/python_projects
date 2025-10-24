import logging
import pandas as pd
import matplotlib.pyplot as plt
from utilsforecast.plotting import plot_series
from neuralforecast.models import *
from neuralforecast.core import NeuralForecast


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# --------------------------------------------------------------

static_df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE_static.csv')
print(f"df[{len(static_df)}, {list(static_df.columns)}")
# print(static_df.head())

df = pd.read_csv(
    'https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE.csv',
    parse_dates=['ds'],
)
print(f"df[{len(df)}, {list(df.columns)}, {df.index[-1]}]")
# print(df.head())

futr_df = pd.read_csv(
    'https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE_futr.csv',
    parse_dates=['ds'],
)
print(f"df[{len(futr_df)}, {list(futr_df.columns)}, {futr_df.index[0]}]")
# print(futr_df.head())



models = [
    # NHITS(h = horizon,
    #         max_steps=3,
    #         input_size = 5*horizon,
    #         futr_exog_list = ['gen_forecast', 'week_day'], # <- Future exogenous variables
    #         hist_exog_list = ['system_load'], # <- Historical exogenous variables
    #         stat_exog_list = ['market_0', 'market_1'], # <- Static exogenous variables
    #         scaler_type = 'robust'),
    # MLP(h = 1,
    #     input_size = 24,
    #     max_steps=10,
    #     # futr_exog_list = ['gen_forecast', 'week_day'], # <- Future exogenous variables
    #     # hist_exog_list = ['system_load'], # <- Historical exogenous variables
    #     hist_exog_list = ['gen_forecast', 'week_day'], # <- Future exogenous variables
    #     stat_exog_list = ['market_0', 'market_1'], # <- Static exogenous variables
    #     scaler_type = 'robust',
    # ),
    NBEATS(h = 1,
        input_size = 24,
        max_steps=10,
        # futr_exog_list = ['gen_forecast', 'week_day'], # <- Future exogenous variables
        # hist_exog_list = ['system_load'], # <- Historical exogenous variables
        hist_exog_list = ['gen_forecast', 'week_day'], # <- Future exogenous variables
        stat_exog_list = ['market_0', 'market_1'], # <- Static exogenous variables
        scaler_type = 'robust',
    ),
]

nf = NeuralForecast(models=models, freq='h')
nf.fit(df=df, static_df=static_df)

try:
    # Y_hat_df = nf.predict(h=len(futr_df), futr_df=futr_df)
    Y_hat_df = nf.predict(df=futr_df)
    Y_hat_df.head()

    ax = plt.gca()
    plot_series(df, Y_hat_df, ax=ax)
    plt.show()
except Exception as e:
    print("ERROR", e)
