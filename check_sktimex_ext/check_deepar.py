
# Example using 'amazon/chronos-t5-tiny' model
from matplotlib import pyplot as plt
from sktime.datasets import load_airline
from neuralforecastx.models.deepar import DeepAR
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = DeepAR(
    h=12,
    input_size=24,
    lstm_n_layers=1,
    # trajectory_samples=100,
    # loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
    # loss={
    #     "clazz": "DistributionLoss",
    #     "distribution": "StudentT",
    #     "level": [80, 90],
    #     "return_params": True
    # },
    # valid_loss=MQLoss(level=[80, 90]),
    # valid_loss={
    #     "clazz": "MQLoss",
    #     "level": [80, 90]
    # },
    # valid_loss="MAE",
    learning_rate=0.005,
    # stat_exog_list=['airline1'],
    # futr_exog_list=['trend'],
    max_steps=100,
    val_check_steps=10,
    early_stop_patience_steps=-1,
    # scaler_type='standard',
    enable_progress_bar=True,
)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

plot_series(y_train, y_test, y_pred, labels=["train", "test", "pred"])
plt.show()
