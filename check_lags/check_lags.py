import numpy as np
from sktimex.transform import LagsTrainTransform, LagsPredictTransform
import numpyx as npx


def main():
    X_all = npx.ij_matrix(200, 4)
    y_all = npx.ij_matrix(200, 1)
    xlags = 0
    ylags = 4
    tlags = 2

    X1 = X_all[:100]
    y1 = y_all[:100]
    X2 = X_all[100:]
    y2 = y_all[100:]
    fh = len(y2)

    tt = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        transpose=False,
        flatten=False,
        concat=None,
        encoder=None,
        decoder=-1,
        recursive=False
    )

    Xt, yt = tt.fit_transform(y1, X1)

    pt = tt.predict_transform()
    y_pred = pt.fit(y1, X1).transform(fh,X2)

    yp = np.zeros((1, 2, 1), dtype=float)

    i = 0
    while i < fh:
        Xp = pt.step(i)

        for j in range(tlags):
            yp[0, j, :] = y2[i+j]

        i = pt.update(i, yp)
    pass


if __name__ == '__main__':
    main()
