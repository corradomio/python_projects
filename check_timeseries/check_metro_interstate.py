import logging.config

import matplotlib.pyplot as plt
import pandasx as pdx
import pandasx_time as pdxt
from numpy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from sktime.utils.plotting import plot_series


def main2():

    # df = pdx.read_data("D:\\Dropbox\\Datasets\\adda\\vw_food_import_aed_pred.csv")
    # valid, nan = pdx.nan_split(df, columns='import_aed')

    df = pdx.read_data("D:\\Dropbox\\Datasets\\other\\Metro_Interstate_Traffic_Volume_valid.csv",
                       onehot=['holiday', 'weather_main'],
                       datetime=('date_time', '%Y-%m-%d %H:%M:%S', 'H'),
                       index=['date_time'],
                       ignore=['date_time', 'holiday', 'weather_main', 'weather_description'],
                       reindex=True)

    df = pdxt.dataframe_periodic(df, freq='D', columns=['cos', 'sin'], method='cossin', datetime=None)

    df.iloc[1:25].plot.scatter('cos', 'sin').set_aspect('equal')
    plt.show()

    X, y = pdx.xy_split(df, target='traffic_volume')

    # plot_series(df['traffic_volume'].iloc[-1000:], labels=['traffic_volume'])
    plot_series(y.iloc[-1000:], labels=['traffic_volume'])

    plt.tight_layout()
    plt.show()

    npy = y['traffic_volume'].iloc[:32768].to_numpy()

    fft_y = fft(npy)
    plt.plot(fft_y)
    plt.show()

    pass
# end


def main1():
    df = pdx.read_data("D:\\Dropbox\\Datasets\\other\\Metro_Interstate_Traffic_Volume_valid.csv",
                       onehot=['holiday', 'weather_main'],
                       datetime=('date_time', '%Y-%m-%d %H:%M:%S', 'H'),
                       index=['date_time'],
                       ignore=['date_time', 'holiday', 'weather_main', 'weather_description'],
                       reindex=True
                       )

    df = pdxt.dataframe_periodic(df, freq='D', columns=['cos', 'sin'], method='cossin', datetime=None)
    n = len(df)

    train_df = df[:int(.7*n)]
    val_df = df[int(.7*n):int(.9*n)]
    test_df = df[int(.9*n):]

    scaler = MinMaxScaler()
    scaler.fit(df)

    train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
    val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
    test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

    train_df.to_csv('./data/train.csv')
    val_df.to_csv('./data/val.csv')
    test_df.to_csv('./data/test.csv')

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main1()
