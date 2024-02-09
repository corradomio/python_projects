import pandas as pd
# from NBEATS import NeuralBeats
#
# data = pd.read_csv('test.csv')
# data = data.values        # (nx1 array)
#
# model = NeuralBeats(data=data, forecast_length=5)
# model.fit()
# forecast = model.predict()

import torch
from nbeatsn import NBeatsNet

nn = NBeatsNet(forecast_length=(3, 1),
               backcast_length=(12, 3))

data = torch.rand((16, 12, 3))

nn(data)

