
# Example using 'amazon/chronos-t5-tiny' model
from matplotlib import pyplot as plt
from sktime.datasets import load_airline
from neuralforecastx.models.fedformer import FEDformer
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = FEDformer(
    h=12,
    input_size=24,
    modes=64,
    hidden_size=64,
    conv_hidden_size=128,
    n_head=8,
    loss="MAE",
    # futr_exog_list=calendar_cols,
    # scaler_type='robust',
    # scaler_type="standard",
    learning_rate=0.005,
    max_steps=100,
    batch_size=2,
    windows_batch_size=32,
    val_check_steps=50,
    early_stop_patience_steps=-1
)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

plot_series(y_train, y_test, y_pred, labels=["train", "test", "pred"])
plt.show()
