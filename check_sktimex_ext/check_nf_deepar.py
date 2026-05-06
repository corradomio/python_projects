import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

from neuralforecastx.models.deepar import DeepARForecastOnly

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

nf = NeuralForecast(
    models=[DeepARForecastOnly(
        h=12,
        input_size=24,
        lstm_n_layers=1,
        trajectory_samples=100,
        loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
        valid_loss=MQLoss(level=[80, 90]),
        learning_rate=0.005,
        stat_exog_list=['airline1'],
        futr_exog_list=['trend'],
        max_steps=100,
        val_check_steps=10,
        early_stop_patience_steps=-1,
        scaler_type='standard',
        enable_progress_bar=True,
        compatibility=True
    ),
    ],
    freq='ME'
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
Y_hat_df = nf.predict(futr_df=Y_test_df)

# Plot quantile predictions
Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['DeepAR-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:],
                 y1=plot_df['DeepAR-lo-90'][-12:].values,
                 y2=plot_df['DeepAR-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
