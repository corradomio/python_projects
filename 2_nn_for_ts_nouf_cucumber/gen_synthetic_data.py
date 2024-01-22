import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandasx as pdx
from math import pi


def gen_data(nseasons, seasonality=12, offset=0., eps=0.):
    nsamples = nseasons*seasonality
    x = np.arange(nsamples)
    t = 2*np.pi*((x + offset)/seasonality)
    y = (np.sin(t) + 1)/2
    if eps > 0:
        y = y + np.random.normal(scale=eps, size=nsamples)
        y[y < 0] = 0.
        y[y > 1] = 1.
    return x, y
# end


def plot_data(noise=False):
    # x, y = gen_data(8*12, offset=3.333, eps=0.0)
    x, y = gen_data(8, offset=pi + pi/4, eps=0.1 if noise else 0.)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(x, y)
    plt.scatter(x, y, c='red')
    # plt.axes().set_aspect('equal')
    # plt.gca().set_aspect(2)
    plt.tight_layout()
    # plt.show()

    fname = 'ts_noisy.jpg' if noise else 'ts_perfect.jpg'
    plt.savefig(fname, dpi=300)

    return x, y


def main():
    x, y0 = plot_data(False)
    _, y1 = plot_data(True)

    dt = pd.date_range(start='2015-01', periods=len(x), freq='MS').map(str).array

    y0 = y0.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    dt = dt.reshape(-1, 1)

    data = np.concatenate([dt, y0, y1], axis=1)

    df = pd.DataFrame(data=data, columns=['Date', 'perfect', 'noisy'])
    pdx.write_data(df, 'perfect_noisy_ts.csv', )
    pass


if __name__ == "__main__":
    main()
