import numpy as np
import pandas as pd


def const_wave(x: np.ndarray, c: float = 0) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(x.shape, dtype=float) + c
    return y, x


def sin_wave(x: np.ndarray, a: float = 1, phase: float = 0) -> tuple[np.ndarray, np.ndarray]:
    y = a * np.sin(2 * x * np.pi + phase) + a*1.1
    return y, x


def sinabs_wave(x: np.ndarray, a: float = 1, phase: float = 0) -> tuple[np.ndarray, np.ndarray]:
    y = a * (2 * np.abs(np.sin(2 * x * np.pi + phase))) + a*0.1
    return y, x


def square_wave(x: np.ndarray, a: float = 1, phase: float = 0) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(x.shape, dtype=float)
    t = (x + phase) % 1
    y[t < 0.5] = a*0.1
    y[t >= 0.5] = +a
    return y, t


def sawtooth_wave(x: np.ndarray, a: float = 1, phase: float = 0) -> tuple[np.ndarray, np.ndarray]:
    # y = np.zeros(x.shape, dtype=float)
    t = (x + phase) % 1
    y = a * t + a*0.1
    return y, x


def inverted_sawtooth_wave(x: np.ndarray, a: float = 1, phase: float = 0) -> tuple[np.ndarray, np.ndarray]:
    # y = np.zeros(x.shape, dtype=float)
    t = (x + phase) % 1
    y = (a - a * t) + a*0.1
    return y, x



def triangle_wave(x: np.ndarray, a: float = 1, phase: float = 0) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(x.shape, dtype=float)
    t = (x + phase) % 1.0
    y[t <= 0.5] = a * (0 + 2 * t[t <= 0.5])
    y[t >= 0.5] = a * (1 + 2* (0.5 - t[t >= 0.5]))
    y += a * 0.1
    return y, x


def noise_signal(y: np.ndarray, noise: float = 0.) -> np.ndarray:
    if noise == 0:
        return y
    k = len(y)
    a = y.max() - y.min()
    if a == 0.: a = 1.
    return np.abs(y + a * noise * np.random.normal(0, 1, size=k))


def add_trend(y: np.ndarray, b0: float, bn: float) -> np.ndarray:
    m = len(y)
    t = np.linspace(b0, bn, num=m, dtype=float)
    return y + t


# ---------------------------------------------------------------------------

def set_dt_index(df: pd.DataFrame, use=False):
    if not use:
        return df

    n = len(df)
    dt_index = pd.date_range("2020-01-01", periods=n, freq="MS")
    dft = df.set_index(dt_index)
    return dft


def create_synthetic_data(n: int = 12 * 7, noise=0., a: float = 1, phase: float = 0) -> pd.DataFrame:
    """
    Note: the reference is 12 months
    The wave must have: 3,6,12,24,36,48
    n: n of points
    noise: quiantity of noise
    a: amplitude of the signal
    phase: phase of the signal
    """
    df_list = []
    m = n+1

    # PERIODS = [3,6,12,24,36,48]
    PERIODS = [3, 6, 12, 24, 48]

    # -- const_wave --
    for c in [1]:
        x = np.linspace(0, 1, num=m, dtype=float)
        y, x = const_wave(x, c)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        if c == 0: name="zero"
        elif c > 0: name="pos"
        elif c < 0: name="neg"
        else: name="unk"
        df["cat"] = f"{name}"
        df_list.append(df)

    # -- square_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = square_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sq{p}"
        df_list.append(df)

    # -- sin_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = sin_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sin{p}"
        df_list.append(df)

    # -- triangle_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = triangle_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"tri{p}"
        df_list.append(df)

    # -- sawtooth_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = sawtooth_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"saw{p}"
        df_list.append(df)

    # -- inverted_sawtooth_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = inverted_sawtooth_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"was{p}"
        df_list.append(df)

    # -- sinabs_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = sinabs_wave(x, a, phase)
        y = noise_signal(y, noise)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sinabs{p}"
        df_list.append(df)

    # -----------------------------------------------------------------------
    # With trend

    # -- const_wave --
    for c in [1]:
        x = np.linspace(0, n, num=m, dtype=float)
        y, x = const_wave(x, c)
        y = noise_signal(y, noise)
        y = add_trend(y, 0, 1)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        if c == 0: name="zero"
        elif c > 0: name="pos"
        elif c < 0: name="neg"
        else: name="unk"
        df["cat"] = f"{name}-t"
        df_list.append(df)

    # -- square_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = square_wave(x, a, phase)
        y = noise_signal(y, noise)
        y = add_trend(y, 0, 1)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sq{p}-t"
        df_list.append(df)

    # -- triangle_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = triangle_wave(x, a, phase)
        y = noise_signal(y, noise)
        y = add_trend(y, 0, 1)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"tri{p}-t"
        df_list.append(df)

    # -- sawtooth_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = sawtooth_wave(x, a, phase)
        y = noise_signal(y, noise)
        y = add_trend(y, 0, 1)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"saw{p}-t"
        df_list.append(df)

    # -- inverted_sawtooth_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = inverted_sawtooth_wave(x, a, phase)
        y = noise_signal(y, noise)
        y = add_trend(y, 0, 1)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"was{p}-t"
        df_list.append(df)

    # -- sin_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = sin_wave(x, a, phase)
        y = noise_signal(y, noise)
        y = add_trend(y, 0, 1)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sin{p}-t"
        df_list.append(df)

    # -- sinabs_wave --
    for p in PERIODS:
        x = np.linspace(0, n, num=m, dtype=float)/p
        y, x = sinabs_wave(x, a, phase)
        y = noise_signal(y, noise)
        y = add_trend(y, 0, 1)
        df = pd.DataFrame(data=np.array([y, x]).T, columns=["y", "x"])
        df["cat"] = f"sinabs{p}-t"
        df_list.append(df)

    # -----------------------------------------------------------------------

    df_list = [
        set_dt_index(df, False)
        for df in df_list
    ]

    # -- concat --
    # df = pd.concat(df_list, axis=0, ignore_index=True)
    df = pd.concat(df_list, axis=0, ignore_index=False)

    return df


# def main():
#     df = create_syntethic_data(12 * 10, 0.0, 1, 0.33)
#     dfdict = pdx.groups_split(df, groups="cat")
#     for g in dfdict:
#         dfg = dfdict[g]
#         skp.plot_series(dfg["y"], title=g)
#         # plt.ylim((-1.2, 1.2))
#         plt.show()
#     # end
#     pass


# if __name__ == "__main__":
#     main()
