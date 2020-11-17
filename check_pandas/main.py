import numpy as np
import pandas as pd

s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
i = s.index
v = s.values

t = pd.Series(np.random.randn(5))

df = pd.DataFrame({
    "s1": s,
    "s2": t
})


print("done")