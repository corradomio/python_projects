from lags_alone import LagsTrainTransform, LagsPredictTransform
import numpy as np


def size_split(*arrys, size=.5) -> list[np.ndarray]:
    def sz(a):
        if size > 1:
            return size
        else:
            return int(size * len(a))

    splits = []
    for a in arrys:
        n = sz(a)
        splits.append(a[:n])
        splits.append(a[n:])
    return splits
# end


def ij_matrix(nrows, ncols, dtype=float):
    def _factor(n):
        f = 10
        l = 1
        while n >= f:
            f *= 10
            l += 1
        return l, f

    # mat = np.zeros((nrows, ncols), dtype=int)
    mat = np.zeros((nrows, ncols), dtype=dtype)
    l, f = _factor(ncols)
    if ncols > 1:
        for r in range(nrows):
            for c in range(ncols):
                # mat[r, c] = r * f + c
                # mat[r, c] = (r + 1) * f + (c + 1)
                mat[r, c] = (r + 1) + round((c + 1)/f, l)
    else:
        for r in range(nrows):
            mat[r, 0] = (r+1)

    return mat
# end


xlags = [1,2]
ylags = [1,2]
tlags = [0,1]


def check_flags(X, y):
    Xp0, yp0, Xf0, yf0 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=False,
        concat=False).fit_transform(y=y, X=X)
    Xp1, yp1, Xf1, yf1 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=True,
        concat=False).fit_transform(y=y, X=X)
    Xp2, yp2, Xf2, yf2 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=True,
        concat=True).fit_transform(y=y, X=X)
    Xp3, yp3, Xf3, yf3 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=True,
        concat='xonly').fit_transform(y=y, X=X)
    Xp4, yp4, Xf4, yf4 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=True,
        concat='all').fit_transform(y=y, X=X)

    Xp5, yp5, Xf5, yf5 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=False,
        concat=False).fit_transform(y=y, X=X)
    Xp6, yp6, Xf6, yf6 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=False,
        concat=True).fit_transform(y=y, X=X)

    Xp10, yp10, Xf10, yf10 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=False,
        concat=False,
        transpose=True).fit_transform(y=y, X=X)
    Xp11, yp11, Xf11, yf11 = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        flatten=False,
        concat=True,
        transpose=True).fit_transform(y=y, X=X)
    pass
# end


def main():
    X = ij_matrix(1000, 10)
    y = ij_matrix(1000, 4)

    check_flags(X, y)
    check_flags(None, y)

    tt = LagsTrainTransform(xlags=xlags, ylags=ylags, tlags=tlags,
                            flatten=True,
                            concat=True)
    Xp, yp, Xf, yf = tt.fit_transform(y=y, X=X)

    N = 800
    fh = 10
    Xp, Xf, yp, yf = size_split(X, y, size=N)
    pt = LagsPredictTransform(xlags=xlags, ylags=ylags, tlags=tlags, flatten=True, concat=True)
    yp = pt.fit(y=yp, X=Xp).transform(fh=fh, X=Xf)
    yf = np.zeros((1, 2, 4))

    i = 0
    while i < fh:
        yts, Xts, Xfs = pt.step(i)
        for j in range(2):
            yf[0, j, :] = y[N+i+j]
        i = pt.update(i, yf)
        pass

    pass


if __name__ == "__main__":
    main()
