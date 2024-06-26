from numpyx.utils import ij_matrix
import sktimex as stx


def main():
    X = +ij_matrix(100, 9)
    y = -ij_matrix(100, 2)

    lin = stx.LinearTrainTransform(slots=[2, 2], tlags=3)
    Xt1, yt1 = lin.fit_transform(X, y)

    cnn = stx.CNNTrainTransform(slots=[2, 2], tlags=3)
    Xt2, yt2 = cnn.fit_transform(X, y)

    c3d = stx.CNNTrainTransform3D(slots=[2, 2], tlags=3)
    Xt3, yt3 = c3d.fit_transform(X, y)

    rnn = stx.RNNTrainTransform(slots=[2, 2], tlags=3)
    Xt4, yt4 = rnn.fit_transform(X, y)

    r3d = stx.RNNTrainTransform3D(slots=[2, 2], tlags=3)
    Xt5, yt5 = r3d.fit_transform(X, y)

    pass


if __name__ == "__main__":
    main()
