from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

#          unix                 date   symbol  ...     close  Volume BTC    Volume USD
# 0  1646092800  2022-03-01 00:00:00  BTC/USD  ...  43312.27   52.056320  2.254677e+06
# 1  1646089200  2022-02-28 23:00:00  BTC/USD  ...  43178.98  106.816103  4.612210e+06
# 2  1646085600  2022-02-28 22:00:00  BTC/USD  ...  42907.32  527.540571  2.263535e+07
# 3  1646082000  2022-02-28 21:00:00  BTC/USD  ...  41659.53   69.751680  2.905822e+06
# 4  1646078400  2022-02-28 20:00:00  BTC/USD  ...  41914.97  247.151654  1.035935e+07

ds = pd.read_csv("data/BTC-Hourly.csv")
pprint(ds.head())

plt.plot(ds['unix'], ds["close"])
plt.show()

