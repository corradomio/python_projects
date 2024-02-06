import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandasx as pdx
from math import pi


def gen_multiple_data(nseasons, seasonality=12, offset=0., eps=0.1):
    nsamples = nseasons * seasonality
    x = np.arange(nsamples)
    t = 2 * np.pi * ((x + offset) / seasonality)
    # rand = np.random.normal(scale=eps, size=nsamples)

    # c)lean
    # n)oise

    # sin(t)
    y0c = (np.sin(t) + 1)/2
    # sin(t) + noise
    y0n = y0c + np.random.normal(scale=eps, size=nsamples)
    # sin(t) + 0.5 sin(2t)
    yd = .5*(np.sin(2*t) + 1)/2     # double
    y1c = (y0c + yd)/1.6
    # sin(t) + 0.5 sin(2t) + noise
    y1n = y1c + np.random.normal(scale=eps, size=nsamples)
    # sin(t) + 0.5 sin(t/2)
    yh = 0.5*(np.sin(0.5*t) + 1)/2  # half
    y2c = (y0c + yh)/1.6
    # sin(t) + 0.5 sin(t/2) + noise
    y2n = y2c + np.random.normal(scale=eps, size=nsamples)
    # sin(t) + 0.25 sin(t/2) + .25 sin(2t)
    y3c = (y0c + .5*yh + .5*yd)/1.6
    # sin(t) + 0.25 sin(t/2) + .25 sin(2t) + noise
    y3n = y3c + np.random.normal(scale=eps, size=nsamples)

    xy = [x, y0c, y0n, y1c, y1n, y2c, y2n, y3c, y3n]
    for i in range(1, len(xy)):
        y = xy[i]
        y[y < 0] = 0.
        y[y > 1] = 1.
    return xy


def gen_data(nseasons, seasonality=12, offset=0., eps=0.):
    nsamples = nseasons*seasonality
    x = np.arange(nsamples)
    t = 2*np.pi*((x + offset)/seasonality)
    y = (np.sin(t) + 1)/2

    # y in range [0,1] instead than [-1, 1]
    if eps > 0:
        y = y + np.random.normal(scale=eps, size=nsamples)
        y[y < 0] = 0.
        y[y > 1] = 1.
    return x, y


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

    fname = 'ts_noisy.jpg' if noise else 'ts_clean.jpg'
    plt.savefig(fname, dpi=300)

    return x, y


def simple_data():
    x, y0 = plot_data(False)
    _, y1 = plot_data(True)

    dt = pd.date_range(start='2015-01', periods=len(x), freq='MS').map(str).array

    y0 = y0.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    dt = dt.reshape(-1, 1)

    data = np.concatenate([dt, y0, y1], axis=1)

    df = pd.DataFrame(data=data, columns=['Date', 'clean', 'noisy'])
    pdx.write_data(df, 'clean_noisy_ts.csv', )


def multiple_data():
    xy = gen_multiple_data(8, offset=pi + pi/4)
    dt = pd.date_range(start='2015-01', periods=len(xy[0]), freq='MS').map(str).array
    dt = dt.reshape(-1, 1)

    for i in range(len(xy)):
        xy[i] = xy[i].reshape(-1, 1)

    data = np.concatenate([dt] + xy[1:], axis=1)

    columns = ['Date', 'y0c', 'y0n', 'y1c', 'y1n', 'y2c', 'y2n', 'y3n', 'y3c']
    df = pd.DataFrame(data=data, columns=columns)
    pdx.write_data(df, 'multiple_ts.csv', )


def plot_multiple():
    df = pdx.read_data("multiple_ts.csv", datetime=('Date', '%Y-%m-%d %H:%M:%S', 'M'),
                   index='Date',
                   ignore='Date')

    for c in df.columns:
        plt.clf()
        plt.figure(figsize=(12, 4))

        points = df[c].to_numpy()
        x = np.arange(len(points))
        plt.plot(x, points)
        plt.scatter(x, points, c='red')
        plt.title(c)
        plt.tight_layout()

        fname = f"ts_{c}.png"
        plt.savefig(fname, dpi=300)
        # plt.show()
    pass


def main():
    # simple_data()
    # multiple_data()
    plot_multiple()
    pass


if __name__ == "__main__":
    main()
