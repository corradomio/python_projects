import sktime
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_lags, plot_series, plot_correlations, plot_interval, plot_windows
import pandasx as pdx


def main():
    df = pdx.read_data("D:\\Dropbox\\Datasets\\other\\Metro_Interstate_Traffic_Volume_valid.csv",
                       onehot=['holiday', 'weather_main'],
                       datetime=('date_time', '%Y-%m-%d %H:%M:%S', 'H'),
                       index=['date_time'],
                       ignore=['date_time', 'holiday', 'weather_main', 'weather_description']
                       )

    X, y = pdx.xy_split(df, target='traffic_volume')

    # plot_series(df['traffic_volume'].iloc[-1000:], labels=['traffic_volume'])
    plot_series(y.iloc[-1000:], labels=['traffic_volume'])

    plt.show()

    pass
# end


if __name__ == "__main__":
    main()
