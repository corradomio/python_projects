from sktimex.forecasting.models import create_forecaster
from stdlib.language import null, true, false
from synth import create_synthetic_data
import pandasx as pdx


def main():
    name = "model"
    jmodel = {
        "class": "sktimex.forecasting.model_selection.ForecastingSkoptSearchCV",
        "n_iter": 16,
        "cv": {
            "class": "sktimex.split.slidingwindow.CountWindowSplitter",
            "n_splits": 6,
            "window_length": 48,
            "fh": 12
        },
        "#return_n_best_forecasters": 4,
        "error_score": "raise",
        "verbose": 1,
        "backend": null,
        "forecaster": {
            "class": "sktimex.forecasting.es_rnn.ESRNNForecaster",
            "pred_len": 6,
            "seasonality_type": "single",
            "+datasets": [
                "was12",
                "sq12",
                "sin12",
                "tri12",
                "saw12-t",
                "saw12",
                "tri12-t",
                "sq12-t",
                "was12-t",
                "sin12-t"
            ],
            "season1_length": 3,
            "window": 6,
            "hidden_size": 5,
            "num_layer": 2,
            "num_epochs": 250,
            "lr": 0.1
        },
        "param_grid": {
            "season1_length": [
                3,
                6,
                12
            ],
            "window": [
                6,
                12
            ],
            "hidden_size": [
                5,
                10
            ],
            "num_layer": [
                2,
                5
            ],
            "num_epochs": [
                250,
                500,
                1000
            ],
            "lr": [
                0.1,
                0.01
            ]
        }
    }
    model = create_forecaster(name, jmodel)

    df = create_synthetic_data(12 * 8, 0.0, 1, 0.33)
    y = pdx.groups_select(df, groups=["cat"], values=["sin12"])["y"]

    model.fit(y)



if __name__ == "__main__":
    main()