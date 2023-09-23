import pandas as pd
import numpy as np


def arange(start, stop=None, step=None, offset=0):
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    l = list(range(start, stop, step))
    return [i+offset for i in l]


data = np.array([
        arange(100, offset=1001),
        arange(100, offset=2001),
        arange(100, offset=3002)
    ]).T

indx = np.array(arange(100))


df1 = pd.DataFrame(data=data, columns=['a', 'b','c'], index=indx)

print(df1.head())
