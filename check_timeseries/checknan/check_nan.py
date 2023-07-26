import pandas as pd
import numpy as np


df = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["one", "two", "three"],
)


df["four"] = "bar"
df["five"] = df["one"] > 0

print(df)
print('---')
df2 = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])

print(df2)
nan = df2['one']['b']
print(nan, type(nan), str(nan))
