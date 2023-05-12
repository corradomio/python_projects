import pandas as pd
import sktime
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_lags, plot_series, plot_correlations, plot_interval, plot_windows
import pandasx as pdx
import pandasx_time as pdxt


def main():

    df = pdx.read_data("D:\\Dropbox\\Datasets\\adda\\vw_food_import_aed_pred.csv")

    valid, nan = pdx.nan_split(df, columns='import_aed')

    df = pdx.read_data("D:\\Dropbox\\Datasets\\other\\Metro_Interstate_Traffic_Volume_valid.csv",
                       onehot=['holiday', 'weather_main'],
                       datetime=('date_time', '%Y-%m-%d %H:%M:%S', 'H'),
                       index=['date_time'],
                       ignore=['date_time', 'holiday', 'weather_main', 'weather_description'],
                       reindex=True
                       )

    df = pdxt.dataframe_periodic(df, freq='D', columns=['cos', 'sin'], method='cossin', datetime=None)

    df.iloc[1:25].plot.scatter('cos', 'sin').set_aspect('equal')
    plt.show()

    X, y = pdx.xy_split(df, target='traffic_volume')

    # plot_series(df['traffic_volume'].iloc[-1000:], labels=['traffic_volume'])
    plot_series(y.iloc[-1000:], labels=['traffic_volume'])

    plt.show()

    pass
# end


if __name__ == "__main__":
    main()
