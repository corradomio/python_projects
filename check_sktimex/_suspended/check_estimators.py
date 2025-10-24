from sktime.utils.estimator_checks import check_estimator
from sktime.forecasting.naive import NaiveForecaster
from sktimex.forecasting import ConstantForecaster
from sktimex.forecasting import ReducerForecaster
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# warnings.simplefilter(action='ignore', category=ConvergenceWarning)


ESTIMATORS = [
    # NaiveForecaster,
    # ConstantForecaster
    ReducerForecaster
]

for est in ESTIMATORS:
    check_estimator(est, tests_to_exclude=[
        "test_predict_interval",
        "test_predict_proba",
        "test_predict_quantiles",
        "test_predict_time_index",
        "test_update_predict_predicted_index",

        'test_persistence_via_pickle',
        'test_save_estimators_to_file',
        'test_hierarchical_with_exogeneous'

    ], verbose=True)
