from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

#       time  JohnsonJohnson
# 0  1960.00            0.71
# 1  1960.25            0.63
# 2  1960.50            0.85
# 3  1960.75            0.44
# 4  1961.00            0.61

ds = pd.read_csv("data/jj.csv")
pprint(ds.head())

plt.plot(ds['time'], ds["JohnsonJohnson"])
plt.show()

