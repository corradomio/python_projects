import pandas as pd
from NBEATS import NeuralBeats

data = pd.read_csv('test.csv')
data = data.values        # (nx1 array)

model = NeuralBeats(data=data, forecast_length=5)
model.fit()
forecast = model.predict()