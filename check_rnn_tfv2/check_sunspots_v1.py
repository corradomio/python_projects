import pandas as pd
import matplotlib.pyplot as plt

print("Hello World")

# ,Year,Month,Day,Date In Fraction Of Year,Number of Sunspots,Standard Deviation,Observations,Indicator

df = pd.read_csv("sunspot_data.csv")
data = df["Number of Sunspots"]

plt.clf()
plt.plot(data)
plt.title("Number of Sunspots")
plt.show()

X = df.values
diff = list()
cycle = 132
for i in range(cycle, len(X)):
    value = X[i] - X[i - cycle]
    diff.append(value)

plt.clf()
plt.plot(diff)
plt.title('Sunspots Dataset Differences')
plt.show()

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# y = np.array(RNN_data.value)
y = data.to_numpy()
model = ARIMA(y, order=(1, 0, 1))  # ARMA(1,1) model
model_fit = model.fit()
print(model_fit.summary())
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

date = df["Date In Fraction Of Year"].to_numpy()
# Actual vs Fitted
cut_t = 30
predictions = model_fit.predict()
plot = pd.DataFrame({'Date': date[cut_t:], 'Actual': abs(y[cut_t:]), "Predicted": predictions[cut_t:]})
plot.plot(x='Date', y=['Actual', 'Predicted'], title='ARMA(1,1) Sunspots Prediction', legend=True)
plt.show()

RMSE = np.sqrt(np.mean(residuals ** 2))
print(RMSE)


dataset = df
split = 0.7
#Split into test and training set (70/20 split)
length = len(dataset)
train_length = round(length*split)
test_length = len(dataset) - train_length
train = dataset[0:train_length,:]
test = dataset[train_length:length,:]


def preprocessing(training_set_scaled, n, train_length, days):
    y_train = []
    for i in range(days, train_length - 1):
        y_train.append(
            training_set_scaled[i, 0])  # note that our predictor variable is in the first column of our array
    y_train = np.array(y_train)

    X = np.zeros(shape=[train_length - (days + 1), days, n], dtype=float)
    for j in range(n):
        for i in range(days, train_length - 1):
            X[i - days, :, j] = training_set_scaled[i - days:i, j]
    return X, y_train


from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

nodes=8


# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = nodes, return_sequences=True, input_shape=(new.shape[1], new.shape[2])))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=nodes, return_sequences=True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=nodes, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=nodes))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units=t)) # this is the output layer so this repersetns a single node with our output value
# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
# Fitting the RNN to the Training set
model = regressor.fit(new, y_train, epochs=50, batch_size=22)



if __name__ == "__main__":
    # main()
    pass
