from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

#          Date       Open       High        Low      Close  Adj Close    Volume
# 0  2004-08-19  50.050049  52.082081  48.028027  50.220219  50.220219  44659096
# 1  2004-08-20  50.555557  54.594597  50.300301  54.209209  54.209209  22834343
# 2  2004-08-23  55.430431  56.796799  54.579578  54.754753  54.754753  18256126
# 3  2004-08-24  55.675674  55.855858  51.836838  52.487488  52.487488  15247337
# 4  2004-08-25  52.532532  54.054054  51.991993  53.053055  53.053055   9188602

ds = pd.read_csv("data/googl.csv")
pprint(ds.head())

plt.plot(ds['Date'], ds["Close"])
plt.show()

