import sktime.utils.plotting
from sktime.datasets import load_shampoo_sales
from sktime.forecasting.model_selection import ForecastingGridSearchCV, ForecastingRandomizedSearchCV
from sktimex.forecasting.model_selection import ForecastingGridSearchCV as ForecastingGridSearchCVX
from sktimex.forecasting.model_selection import ForecastingRandomizedSearchCV as ForecastingRandomizedSearchCVX
from sktimex.forecasting.model_selection import ForecastingSkoptSearchCV as ForecastingSkoptSearchCVX
from sktimex.forecasting.model_selection import ForecastingHyperactiveSearchCV as ForecastingHyperactiveSearchCVX
from sktimex.forecasting.model_selection import ForecastingOptunaSearchCV as ForecastingOptunaSearchCVX
from sktime.split import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
import matplotlib.pyplot as plt

y = load_shampoo_sales()

fh = [1,2,3]
# cv = ExpandingWindowSplitter(fh=fh)
# forecaster = NaiveForecaster()
# param_grid = {"strategy" : ["last", "mean", "drift"]}

# gscv = ForecastingGridSearchCV(
#     forecaster=NaiveForecaster(),
#     param_grid={
#         "strategy" : ["last", "mean", "drift"]
#     },
#     cv=ExpandingWindowSplitter(fh=fh),
#     backend_params={
#         "n_jobs": 2
#     },
#     verbose=1
# )
# gscv.fit(y)
# y_pred = gscv.predict(fh)
# sktime.utils.plotting.plot_series(y, y_pred, labels=["y", "y_pred"])
# plt.show()


def check_fgs():
    gscvx = ForecastingGridSearchCVX(
        forecaster={
            "class": "sktime.forecasting.naive.NaiveForecaster"
        },
        param_grid={
            "strategy": ["last", "mean", "drift"]
        },
        cv={
            "class": "sktime.split.ExpandingWindowSplitter",
            "fh": fh
        },
        backend_params={
            "n_jobs": 2
        },
        verbose=1
    )

    gscvx.fit(y)
    y_predx = gscvx.predict(fh)

    sktime.utils.plotting.plot_series(y, y_predx, labels=["y", "y_predx"])
    plt.show()


def check_frs():
    gscvx = ForecastingRandomizedSearchCVX(
        forecaster={
            "class": "sktime.forecasting.naive.NaiveForecaster"
        },
        param_grid={
            "strategy": ["last", "mean", "drift"]
        },
        cv={
            "class": "sktime.split.ExpandingWindowSplitter",
            "fh": fh
        },
        backend_params={
            "n_jobs": 2
        },
        verbose=1
    )

    gscvx.fit(y)
    y_predx = gscvx.predict(fh)

    sktime.utils.plotting.plot_series(y, y_predx, labels=["y", "y_predx"])
    plt.show()


def check_fss():
    gscvx = ForecastingSkoptSearchCVX(
        forecaster={
            "class": "sktime.forecasting.naive.NaiveForecaster"
        },
        param_grid={
            "strategy": ["last", "mean", "drift"]
        },
        cv={
            "class": "sktime.split.ExpandingWindowSplitter",
            "fh": fh
        },
        backend_params={
            "n_jobs": 2
        },
        verbose=1
    )

    gscvx.fit(y)
    y_predx = gscvx.predict(fh)

    sktime.utils.plotting.plot_series(y, y_predx, labels=["y", "y_predx"])
    plt.show()


def check_fhs():
    gscvx = ForecastingHyperactiveSearchCVX(
        forecaster={
            "class": "sktime.forecasting.naive.NaiveForecaster"
        },
        param_grid={
            "strategy": ["last", "mean", "drift"]
        },
        cv={
            "class": "sktime.split.ExpandingWindowSplitter",
            "fh": fh
        },
        backend_params={
            "n_jobs": 2
        },
        verbose=1
    )

    gscvx.fit(y)
    y_predx = gscvx.predict(fh)

    sktime.utils.plotting.plot_series(y, y_predx, labels=["y", "y_predx"])
    plt.show()


def check_fos():
    gscvx = ForecastingOptunaSearchCVX(
        forecaster={
            "class": "sktime.forecasting.naive.NaiveForecaster"
        },
        param_grid={
            "strategy": ["last", "mean", "drift"]
        },
        cv={
            "class": "sktime.split.ExpandingWindowSplitter",
            "fh": fh
        },
        backend_params={
            "n_jobs": 2
        },
        verbose=1
    )

    gscvx.fit(y)
    y_predx = gscvx.predict(fh)

    sktime.utils.plotting.plot_series(y, y_predx, labels=["y", "y_predx"])
    plt.show()



def main():
    # check_fgs()
    # check_frs()
    # check_fss()
    # check_fhs()
    check_fos()
    pass


if __name__ == "__main__":
    main()