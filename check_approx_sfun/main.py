import numpy as np
import numpy.random as npr
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import mean_squared_error
from keras.losses import mean_squared_error


def main():
    N = 10000
    X = np.zeros((N, 2))
    x = npr.rand(10000)
    X[:, 0] = x
    X[:, 1] = x*x
    y = X[:, 0] + X[:, 1]

    model = Sequential()
    model.add(Dense(1, input_shape=(2,)))
    model.summary()

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])

    model.fit(X, y, batch_size=100, epochs=200, verbose=1, validation_split=.2, shuffle=True)

    print(model.predict(np.array([[0.5, 0.25]])))

    x = npr.rand(100).sort()
    y = model.predict()

    pass


if __name__ == "__main__":
    main()
