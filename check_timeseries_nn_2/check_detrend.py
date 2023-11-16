import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series
from pandasx.preprocessing import DetrendTransform, SeasonalityTransform
from scipy import signal


def main():
    x = np.arange(100)
    # y = 20 + 0.8*x + 5.*np.sin(x*np.pi/8)
    y = 5. * np.sin(np.pi/4 + x * np.pi / 8)

    plt.plot(x, y)
    plt.show()

    df = pd.DataFrame(data=y, columns=["y"])

    # plot_series(df['y'], labels=['y'])
    # plt.show()

    train = df[0:80]
    test = df[80:]

    dtt = DetrendTransform(columns="y")
    train_t = dtt.fit_transform(train)
    test_t = dtt.transform(test)

    # signal.detrend()

    # plot_series(train_t['y'], test_t['y'], labels=['train', 'test'])
    # plt.show()

    # test_o = dtt.inverse_transform(test_t)
    # plot_series(train['y'], test_o['y'], labels=['train', 'test'])
    # plt.show()

    st = SeasonalityTransform('y')
    train_s = st.fit_transform(train)
    test_s = st.transform(test)

    plot_series(train_s['y'], test_s['y'], labels=['train', 'test'])
    plt.show()

    pass



if __name__ == "__main__":
    main()
