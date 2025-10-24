from sktimex.transform.lags import matrix, yx_lags, t_lags
from sktimex.transform.lags import yx_lags, t_lags
from sktimex.transform.lint import LinearTrainTransform, LinearPredictTransform
from sktimex.transform.lagt import LagsTrainTransform, LagsPredictTransform

def main():
    X0 = matrix(9, 9)
    y0 = matrix(9, 0)
    X1 = matrix(9, 9, 10)
    y1 = matrix(9, 1, 10)

    lagtt = LagsTrainTransform(xlags=[0], ylags=[0], tlags=[1], flatten=False)

    lintt = LinearTrainTransform(xlags=[1], ylags=[1], tlags=[0])

    Xt1, yt1 = lagtt.fit_transform(y=y0, X=X0)
    Xt2, yt2 = lintt.fit_transform(y=y0, X=X0)

    pass



if __name__ == "__main__":
    main()
