from sktime.forecasting.base import BaseForecaster as BaseForecaster
from ..forecasting.skx.reducer import ReducerForecaster
from ..forecasting.skx.regressor import RegressorForecaster
from ..utils import import_from, dict_exclude, dict_select


def create_forecaster(model_name: str, model_config: dict) -> BaseForecaster:
    assert isinstance(model_name, str)
    assert isinstance(model_config, dict)
    assert "class" in model_config, "Missing mandatory 'class' key in model_config"

    class_name: str = model_config["class"]
    if class_name.startswith("sktime.") or class_name.startswith("sktimex."):
        forecaster = _create_sktime_forecaster(model_name, model_config)
    elif class_name.startswith("sklearn.") or class_name.startswith("sklearnx."):
        forecaster = _create_sklearn_forecaster(model_name, model_config)
    else:
        forecaster = _create_other_forecaster(model_name, model_config)

    return forecaster
# end


def _create_sktime_forecaster(model_name: str, model_config: dict) -> BaseForecaster:
    class_name = model_config["class"]
    model_config = dict_exclude(model_config, ["class"])

    class_instance = import_from(class_name)

    forecaster = class_instance(**model_config)
    return forecaster


def _create_other_forecaster(model_name: str, model_config: dict) -> BaseForecaster:
    return _create_sktime_forecaster(model_name, model_config)


def _create_sklearn_forecaster(model_name: str, model_config: dict) -> BaseForecaster:
    if "window_length" in model_config:
        forecaster = _create_reducer_forecaster(model_name, model_config)
    elif "lags" in model_config:
        forecaster = _create_regressor_forecaster(model_name, model_config)
    else:
        raise ValueError(f"Unsupported model {model_name}: {model_config}")
    return forecaster


REDUCER_PARAMETERS = ["strategy", "window_length", "prediction_length", "scitype", "transformers", "pooling", "windows_identical", "debug"]

def _create_reducer_forecaster(model_name: str, model_config: dict):
    class_name = model_config["class"]
    model_config = dict_exclude(model_config, ["class"])

    assert "window_length" in model_config, "Missing mandatory 'window_length' key in model_config"
    # assert "prediction_length" in model_config, "Missing mandatory 'prediction_length' key in model_config"

    reducer_config = dict_select(model_config, REDUCER_PARAMETERS)
    model_config = dict_exclude(model_config, REDUCER_PARAMETERS)

    forecaster = ReducerForecaster(
        estimator=class_name,
        estimator_args=model_config,
        **reducer_config
    )
    return forecaster



REGRESSOR_PARAMETERS = ["lags", "tlags", "flatten", "debug"]

def _create_regressor_forecaster(model_name: str, model_config: dict):
    class_name = model_config["class"]
    model_config = dict_exclude(model_config, ["class"])

    assert "lags"  in model_config, "Missing mandatory 'lags' key in model_config"
    # assert "tlags" in model_config, "Missing mandatory 'tlags' key in model_config"

    regressor_config = dict_select(model_config, REGRESSOR_PARAMETERS)
    model_config = dict_exclude(model_config, REGRESSOR_PARAMETERS)

    forecaster = RegressorForecaster(
        estimator=class_name,
        estimator_args=model_config,
        **regressor_config
    )
    return forecaster



