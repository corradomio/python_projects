import numpy as np
import pandas as pd
import sktime as skt
import sklearn as skl
from functools import lru_cache


@lru_cache(typed=True)
def fun(arg):
    print("called fun:", arg)
    return arg



def main():
    r = np.arange(10)
    p = pd.DataFrame()
    # print(hash(r))
    print(hash(p))
    for i in range(10):
        fun(1)
        fun("ciccio")
        # fun(r)
        fun(p)
    pass


if __name__ == "__main__":
    main()

