from sktime.forecasting.base import BaseForecaster as BaseForecaster
from ..forecasting.reducer import ReducerForecaster
from ..forecasting.regressor import RegressorForecaster
from ..utils import import_from, dict_exclude, dict_select, create_from



def _startswith(name: str, prefixes: list[str]) -> bool:
    for p in prefixes:
        if name.startswith(p):
            return True
    return False
# end


def create_forecaster(model_name: str, model_config: dict) -> BaseForecaster:
    assert isinstance(model_name, str)
    assert isinstance(model_config, dict)
    assert "class" in model_config, "Missing mandatory 'class' key in model_config"

    class_name: str = model_config["class"]
    if _startswith(class_name, ["sktime.", "sktimex.", "sktimexnn.", "sktimext."]):
        forecaster = _create_sktime_forecaster(model_name, model_config)
    elif _startswith(class_name, ["sklearn.", "sklearnx."]):
        forecaster = _create_sklearn_forecaster(model_name, model_config)
    else:
        forecaster = _create_other_forecaster(model_name, model_config)

    return forecaster
# end


def _create_sktime_forecaster(model_name: str, model_config: dict) -> BaseForecaster:
    # class_name = model_config["class"]
    # model_config = dict_exclude(model_config, ["class"])
    #
    # class_instance = import_from(class_name)
    # forecaster = class_instance(**model_config)

    forecaster = create_from(model_config)

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
        estimator={
            "class": class_name,
        } | model_config,
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
        estimator={
            "class": class_name
        } | model_config,
        **regressor_config
    )
    return forecaster



