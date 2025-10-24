from sktimex.transform.lags import dataframe
from sktimex.transform.lint import LinearTrainTransform


def main():
    X0 = dataframe(9, 9, name="x")
    y0 = dataframe(9, 0, name="y")
    X1 = dataframe(9, 9, 10, name="x")
    y1 = dataframe(9, 1, 10, name="y")

    # lagtt = LagsTrainTransform(xlags=[0], ylags=[0], tlags=[1])
    # Xt1, yt1 = lagtt.fit_transform(y=y0, X=X0)

    lintt = LinearTrainTransform(ylags=[1,3], xlags=[0,2], tlags=[2,4])
    linpt = lintt.predict_transform()

    Xt2, yt2 = lintt.fit_transform(y=y0, X=X0)

    yp = linpt.fit(y=y0, X=X0).transform(fh=len(y1), X=X1)

    pass



if __name__ == "__main__":
    main()
