import numpy as np
import sklearn as skl
import sklearn.linear_model as skll
import matplotlib.pyplot as plt


def main():
    x = np.arange(0, 100, .01)
    y = x + 3*np.random.normal(size=len(x))

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    lr = skll.LinearRegression()
    lr.fit(x, y)
    z = lr.predict(x)

    plt.plot(x, y)
    plt.plot(x, z)
    plt.show()

    


if __name__ == "__main__":
    main()
