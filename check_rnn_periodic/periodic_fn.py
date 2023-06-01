#
# (a0 + b0 t) + SUM(i, (ai + bi t)*sin(ci + di t))
#
# where
#   (a0 + b0 t)     trend
#   (ai + bi t)     amplitude
#   (ci + di t)     seasonality
#
# Seasonality:
#
#   s:      m,h,D,W,M,Q,Y
#   m:      h,D,W,M,Q,Y
#   h:      D,W,M,Q,Y
#   D:      W,M,Q,Y
#   M:      Q,Y
#   Q:      Y
import numpy as np


def periodic_fn(t: np.ndarray, ab=(0, 0), abcd=(0, 1, 0, 1/12)) -> np.ndarray:
    """
        y = (a0 + b0 t) + SUM(i, (ai + bi t)*sin(ci + di t))

    :param t: time
    :param ab: (a0, b0)
    :param abcd: [(a1, b1, c1, d1), ...]
    :return:
    """
    if isinstance(abcd[0], (int, float)):
        abcd = [abcd]

    a, b = ab
    y = a + b*t

    n = len(abcd)
    for i in range(n):
        a, b, c, d = abcd[i]
        y += (a + b*t)*np.sin(c + d*2*np.pi*t)
    return y
