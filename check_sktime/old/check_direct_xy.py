import matplotlib.pyplot as plt
import pandas as pd
import pandasx as pdx
from sklearn.neighbors import KNeighborsRegressor
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series
from sktimex import ScikitForecastRegressor, LinearForecastRegressor


# data loading for illustration (see section 1 for explanation)
# y = load_airline()
dfall = pdx.read_data("vw_food_import_train_test_newfeatures.csv",
                       datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),
                       onehot=['imp_month'],
                       ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country',
                               'producer_price_tonne_src_country',
                               "crude_oil_price", "sandp_500_us", "sandp_sensex_india", "shenzhen_index_china",
                               "nikkei_225_japan",
                               'max_temperature', 'min_temperature'],
                       numeric=['evaporation', 'mean_temperature', 'rainy_days', 'vap_pressure'],
                       # periodic=('imp_date', 'M'),
                       na_values=['(null)'])

dfg = pdx.groups_split(dfall, groups='item_country', drop=True)
df = list(dfg.values())[0]
df = pdx.set_index(df, 'imp_date', drop=True)

X, y = pdx.xy_split(df, target='import_kg')


X_train, X_test_true, y_train, y_test_true = temporal_train_test_split(X, y, test_size=26)
X_test, X_true, y_test, y_true = temporal_train_test_split(X_test_true, y_test_true, train_size=12)
fh = ForecastingHorizon(y_test.index)
fh_test_true = ForecastingHorizon(y_test_true.index)

# forecaster = ScikitForecastRegressor(window_length=15,
#                                      #class_name='sklearn.neighbors.KNeighborsRegressor',
#                                      # n_neighbors=1,
#                                      strategy="direct")

forecaster = LinearForecastRegressor(lags=15)


forecaster.fit(y=y_train, X=X_train, fh=fh)
# 1958/01 - 1960/12
y_pred = forecaster.predict(fh=fh_test_true, X=X_test_true)
plot_series(y_train, y_test_true, y_pred, labels=["y_train", "y_test", "y_pred"], title="direct")
plt.show()





