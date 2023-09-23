import numpy as np
from numpyx.scaler import MinMaxScaler

a = np.array([1, 3, 2, 4, 3, 5])

mms = MinMaxScaler()

b = mms.fit_transform(a)
c = mms.inverse_transform(b)


