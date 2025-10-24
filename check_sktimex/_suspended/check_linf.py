from sktimex.forecasting.regressor import RegressorForecaster
from sktimex.transform.lags import matrix


def main():
    X0 = matrix(9, 9)
    y0 = matrix(9, 0)
    X1 = matrix(9, 9, 10)
    y1 = matrix(9, 1, 10)

    # X0 = None
    # X1 = None

    rf = RegressorForecaster(
        lags=[0, [0]],
        tlags=[0]
    )

    rf.fit(y=y0, X=X0)
    print(rf.predict_history(fh=5, X=X1, yh=y0, Xh=X0))

    pass



if __name__ == "__main__":
    main()
