import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.utils import plot_series


def main():
    # the sequence [0,1,.....99]
    data = np.arange(100)
    # just to assing a datetime index
    index = pd.period_range(start='2020-01-01', periods=100, freq='D')

    # converted in a dataframe to have an index
    df = pd.DataFrame(data=data, columns=['y'], index=index)
    # trick to draw plots correctly
    df['z'] = 0
    y = df[['y']]

    plot_series(df['y'], labels=['y'])
    plt.show()

    train = y.iloc[0:50]
    skip = y.iloc[50:75]
    predict = y.iloc[75:]

    # it is used the
    forecaster = make_reduction(sklearn.linear_model.LinearRegression())
    forecaster.fit(y=train)

    # this is the PAST data saved in the training
    y = forecaster._y
    X = forecaster._X
    print("model      y:", None if y is None else list(y[0:4]))
    print("model      X:", None if X is None else list(X[0:4]))
    print("model cutoff:", forecaster.cutoff)
    print("model     fh:", forecaster._fh)

    # predictions AFTER the skipped period
    fh = ForecastingHorizon(predict.index, is_relative=False)
    y_pred = forecaster.predict(fh=fh)

    # It works BECAUSE it generates 'recursively' ALL future values from the 'cutoff'
    # to the end of the forecasting horizon, then, it return just the values in the
    # selected forecasting horizon
    plot_series(train['y'], y_pred['y'], df['z'], labels=['train', 'pred', 'zero'])
    plt.show()

    # we suppose to delete X and y, that is, to remove the data used in the training
    # this is an 'hack', because there is no official cleaning API
    forecaster._y = None
    forecaster._X = None

    # we try to predict the forecasting horizon
    try:
        y_pred = forecaster.predict(fh=fh)
    except Exception as e:
        exc = traceback.format_exc()
        print(f"It is not able to compute the prediction: {e}\n{exc}")

    # Note: the exception is:
    #
    # Traceback (most recent call last):
    #   File "D:\Projects.github\python_projects\check_sktime\check_forecast.py", line 47, in main
    #     y_pred = forecaster.predict(fh=fh)
    #   File "D:\Python\Anaconda3-2022.05\envs\ipredict\lib\site-packages\sktime\forecasting\base\_base.py", line 438, in predict
    #     y_pred = self._predict(fh=fh, X=X_inner)
    #   File "D:\Python\Anaconda3-2022.05\envs\ipredict\lib\site-packages\sktime\forecasting\base\_sktime.py", line 30, in _predict
    #     y_pred = self._predict_fixed_cutoff(
    #   File "D:\Python\Anaconda3-2022.05\envs\ipredict\lib\site-packages\sktime\forecasting\base\_sktime.py", line 74, in _predict_fixed_cutoff
    #     y_pred = self._predict_last_window(
    #   File "D:\Python\Anaconda3-2022.05\envs\ipredict\lib\site-packages\sktime\forecasting\compose\_reduce.py", line 864, in _predict_last_window
    #     y_last, X_last = self._get_last_window()
    #   File "D:\Python\Anaconda3-2022.05\envs\ipredict\lib\site-packages\sktime\forecasting\base\_sktime.py", line 137, in _get_last_window
    #     y = self._y.loc[start:cutoff].to_numpy()
    # AttributeError: 'NoneType' object has no attribute 'loc'
    #
    # the important is
    #
    #     y = self._y.loc[start:cutoff].to_numpy()
    #
    # that is: the sktime implementation USES '_y', the PAST data to compute the future data.
    #

    # Now, it is possible to 'update' the PAST data
    forecaster.update(y=skip, X=None, update_params=False)

    # and to generate the correct predictions
    y_pred = forecaster.predict(fh=fh)
    plot_series(train['y'], skip['y'], y_pred['y'], df['z'], labels=['train', 'skip', 'pred', 'zero'])
    plt.show()

    pass


if __name__ == "__main__":
    main()
