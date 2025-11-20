import numpy as np
import pandas as pd
import pandasx as pdx
from math import pi
import sktime.utils.plotting as skp
import matplotlib.pyplot as plt


def const_wave(x: np.ndarray, c: float=0) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(x.shape, dtype=float) + c
    return y, x


def sin_wave(x: np.ndarray, a: float=1, phase: float=0) -> tuple[np.ndarray, np.ndarray]:
    y = a*np.sin(x*np.pi + phase)
    return y, x


def sinabs_wave(x: np.ndarray, a: float=1, phase: float=0) -> tuple[np.ndarray, np.ndarray]:
    y = a*(2*np.abs(np.sin(x*np.pi + phase))-1)
    return y, x


def square_wave(x: np.ndarray, a: float=1, phase: float=0) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(x.shape, dtype=float)
    t = (x + phase) % 1
    y[t< 0.5] = -a
    y[t>=0.5] = +a
    return y, t


def triangle_wave(x: np.ndarray, a: float=1, phase: float=0) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(x.shape, dtype=float)
    t = (x + phase) % 1
    y = a*t
    return y, x


def noise_signal(y: np.ndarray, noise: float=0.) -> np.ndarray:
    if noise == 0:
        return y
    k = len(y)
    a = y.max()-y.min()
    if a == 0.: a = 1.
    return y + a*noise*np.random.normal(0, 1, size=k)


def set_dt_index(df: pd.DataFrame, use=False):
    if not use:
        return df

    n = len(df)
    dt_index = pd.date_range("2020-01-01", periods=n, freq="MS")
    dft = df.set_index(dt_index)
    return dft




def create_syntethic_data(n: int=12*7, noise=0., a:float=1, phase: float=0) -> pd.DataFrame:
    df_list = []

    # -- const --
    # const 0
    for c in [-1, 0, 1]:
        x = np.linspace(0, 1, num=n, dtype=float)
        y, x = const_wave(x, c)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"c={c}"
        df_list.append(df)

    # -- square --
    # square 1
    for c in [1,2,4,8]:
        x = np.linspace(0,c, num=n, dtype=float)
        y, x = square_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sq{c}"
        df_list.append(df)

    # -- sin --
    # sin 1
    for c in [1,2,4,8]:
        x = np.linspace(0, c, num=n, dtype=float)
        y, x = sin_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sin{c}"
        df_list.append(df)

    # -- triangle --
    # triangle 1
    for c in [1, 2, 4, 8]:
        x = np.linspace(0, c, num=n, dtype=float)
        y, x = triangle_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"tri{c}"
        df_list.append(df)

    # -- sin --
    # sinabs 1
    for c in [1,2,4,8]:
        x = np.linspace(0, c, num=n, dtype=float)
        y, x = sinabs_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sinabs{c}"
        df_list.append(df)

    df_list = [
        set_dt_index(df, False)
        for df in df_list
    ]

    # -- concat --
    # df = pd.concat(df_list, axis=0, ignore_index=True)
    df = pd.concat(df_list, axis=0, ignore_index=False)

    return df

def main():
    df = create_syntethic_data(12*7, 0.0, 1, 0.33)
    dfdict = pdx.groups_split(df, groups="cat")
    for g in dfdict:
        dfg = dfdict[g]
        skp.plot_series(dfg["y"], title=g)
        # plt.ylim((-1.2, 1.2))
        plt.show()
    # end
    pass
# end


if __name__ == "__main__":
    main()

