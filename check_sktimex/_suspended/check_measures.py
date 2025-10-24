from sklearn.metrics import r2_score
from sklearnx.metrics import weighted_absolute_percentage_error
from sktime.datasets import load_airline
from sktime.split import temporal_train_test_split
from sktimex.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktimex.forecasting import create_forecaster


def main():
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)

    forecaster = create_forecaster("lin", {
        "class": "sklearn.linear_model.LinearRegression",
        # "window_length": 12,
        # "prediction_length": 1
        "lags": 12,
        "tlags": 1
    })

    fh = ForecastingHorizon(y_test.index)

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    print(mean_absolute_percentage_error(y_test, y_pred))
    print(weighted_absolute_percentage_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))

if __name__ == "__main__":
    main()

