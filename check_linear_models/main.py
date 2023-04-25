from sklearn.linear_model import LinearRegression
from etime.linear_model import LinearModel
from pandas import DataFrame


def main():

    lr = LinearRegression()
    lm = LinearModel(class_name='sklearn.linear_model.LinearRegression', lag=5)

    df = DataFrame({'x': list(range(100)), 'y': [(i-1+i/100) for i in range(100)]})
    X = df[['x']]
    y = df['y']

    lr.fit(X, y)

    lm.fit(X, y)
    pass


if __name__ == "__main__":
    main()
