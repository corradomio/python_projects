import pandas as pd
import pandasx as pdx
import numpy as np
import numpyx as npx
from sktime.forecasting.base import ForecastingHorizon

from stdlib import qualified_name
from sklearn.linear_model import LinearRegression

from etime.linear_model import LinearForecastRegressor
from etime.scikit_model import ScikitForecastRegressor


def main():
    df = pdx.read_data('stallion_all.csv', datetime=('date', '%Y-%m-%d'), index='date')
    df_groups = pdx.dataframe_split_on_groups(df, ['agency', 'sku'])

    for g in df_groups:
        dfg = df_groups[g]
        dfg.to_csv(f'stallion-{".".join(g)}.csv')
        dfg = dfg.to_period('M')
        dfg = dfg[dfg.columns.difference(['agency', 'sku', 'date', 'timeseries'])]

        # -------------------------------------------------------------------

        n = len(dfg)
        t = int(0.8*n)

        x = dfg[dfg.columns.difference(['volume'])]
        y = dfg['volume']

        x.to_csv(f'x.csv', index=False, header=False)
        y.to_csv(f'y.csv', index=False, header=False)

        x_train, x_test = x[:t], x[t:]
        y_train, y_test = y[:t], y[:t]
        fh = ForecastingHorizon(x_test.index, is_relative=False)

        # -------------------------------------------------------------------

        skr = ScikitForecastRegressor(
            class_name=qualified_name(LinearRegression),
            window_length=1,
            strategy='recursive'
        )
        skr.fit(X=x_train, y=y_train, fh=fh)
        y_pred_1 = skr.predict(X=x_test, fh=fh)

        lfr = LinearForecastRegressor(
            class_name=qualified_name(LinearRegression),
            lag=dict(
                input=1,
                target=1,
                current=False
            )
        )
        lfr.fit(X=x_train, y=y_train)
        y_pred_2 = lfr.predict(X=x_test, fh=fh)

        print(g)
# end


if __name__ == "__main__":
    main()
