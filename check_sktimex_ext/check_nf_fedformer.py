import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import FEDformer
from neuralforecast.utils import AirPassengersPanel, augment_calendar_df

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = FEDformer(
    h=12,
    input_size=24,
    modes=64,
    hidden_size=64,
    conv_hidden_size=128,
    n_head=8,
    loss=MAE(),
    futr_exog_list=calendar_cols,
    scaler_type='robust',
    learning_rate=1e-3,
    max_steps=500,
    batch_size=2,
    windows_batch_size=32,
    val_check_steps=50,
    early_stop_patience_steps=2
)

nf = NeuralForecast(
    models=[model],
    freq='ME',
)
nf.fit(df=Y_train_df, static_df=None, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

if model.loss.is_distribution_output:
    print("is_distribution_output")
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['FEDformer-median'], c='blue', label='median')
    plt.fill_between(x=plot_df['ds'][-12:],
                    y1=plot_df['FEDformer-lo-90'][-12:].values,
                    y2=plot_df['FEDformer-hi-90'][-12:].values,
                    alpha=0.4, label='level 90')
    plt.grid()
    plt.legend()
    plt.plot()
else:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['FEDformer'], c='blue', label='Forecast')
    plt.legend()
    plt.grid()

plt.show()