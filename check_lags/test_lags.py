import numpy as np
import numpyx as npx
from sktimex import LagsTrainTransform
from is_instance import is_instance

MX = 4
MY = 1

XNONE = 0
XLAGS = 4
YLAGS = 4
TLAGS = 2

X_all = npx.ij_matrix(200, MX)
y_all = npx.ij_matrix(200, MY)

Xt = X_all[:100]    # train
yt = y_all[:100]    # train
Xp = X_all[100:]    # predict
yp = y_all[100:]    # predict
fh = len(yp)


# def test_1t():
#     xlags = XNONE
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=None,
#         encoder=None,
#         decoder=None,
#         recursive=False
#     )
#
#     X_past, y_future = tt.fit_transform(y=yt, X=Xt)
#
#     assert isinstance(X_past, np.ndarray)
#     assert isinstance(y_future, np.ndarray)
#     assert X_past.shape == (94, ylags, MY)
#     assert y_future.shape == (94, tlags, MY)
#
#     assert X_past[0, 0, 0] == 1.
#     assert y_future[0, 0, 0] == 5.


# def test_1p():
#     xlags = XNONE
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=None,
#         encoder=None,
#         decoder=None,
#         recursive=False
#     )
#
#     pt = tt.predict_transform()
#
#     y_pred = pt.fit(y=yt, X=Xt).transform(fh=fh,X=Xp)
#     X_pred = pt.step(0)
#
#     assert is_instance(y_pred, np.ndarray)
#     assert is_instance(X_pred, np.ndarray)
#     assert y_pred.shape == (fh, MY)
#     assert X_pred.shape == (1, ylags, MY)
#     assert X_pred[0, 0, 0] == 97.


# def test_2():
#     xlags = XNONE
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=True,    # changed
#         encoder=None,
#         decoder=None,
#         recursive=False
#     )
#
#     X_past, y_future = tt.fit_transform(y=yt, X=Xt)
#
#     assert X_past.shape == (94, ylags, MY)
#     assert y_future.shape == (94, tlags, MY)


# def test_3():
#     xlags = XLAGS           # changed
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=False,    # changed
#         encoder=None,
#         decoder=None,
#         recursive=False
#     )
#
#     X_past, y_future = tt.fit_transform(y=yt, X=Xt)
#
#     assert X_past.shape == (94, ylags, MY)
#     assert y_future.shape == (94, tlags, MY)


# def test_4():
#     xlags = XLAGS           # changed
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=True,    # changed
#         encoder=None,
#         decoder=None,
#         recursive=False
#     )
#
#     X_past, y_future = tt.fit_transform(y=yt, X=Xt)
#
#     assert X_past.shape == (94, ylags, MX_MY)
#     assert y_future.shape == (94, tlags, MY)


# def test_5():
#     xlags = XLAGS           # changed
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=True,    # changed
#         encoder=0,
#         decoder=None,
#         recursive=False
#     )
#
#     X_past, y_encoder_future = tt.fit_transform(y=yt, X=Xt)
#
#     assert isinstance(y_encoder_future, tuple) and len(y_encoder_future) == 2
#     y_encoder, y_future = y_encoder_future
#
#     assert X_past.shape == (94, ylags, MX+MY)
#     assert y_encoder.shape == (94, ylags, MY)
#     assert y_future.shape == (94, tlags, MY)
#
#     assert y_encoder[0,0,0] == 1.


# def test_6():
#     xlags = XLAGS           # changed
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=True,    # changed
#         encoder=None,
#         decoder=0,
#         recursive=False
#     )
#
#     X_past_future, y_future = tt.fit_transform(y=yt, X=Xt)
#
#     assert isinstance(X_past_future, tuple) and len(X_past_future) == 2
#     X_past, X_future = X_past_future
#
#     assert X_past.shape == (94, ylags, MX+MY)
#     assert X_future.shape == (94, tlags, MX)
#     assert y_future.shape == (94, tlags, MY)
#
#     assert X_future[0,0,0] == 5.1


# def test_7():
#     xlags = XLAGS           # changed
#     ylags = YLAGS
#     tlags = TLAGS
#
#     tt = LagsTrainTransform(
#         xlags=xlags, ylags=ylags, tlags=tlags,
#         transpose=False,
#         flatten=False,
#         concat=True,    # changed
#         encoder=None,
#         decoder="past",
#         recursive=False
#     )
#
#     X_past_future, y_future = tt.fit_transform(y=yt, X=Xt)
#
#     assert isinstance(X_past_future, tuple) and len(X_past_future) == 2
#     X_past, X_future = X_past_future
#
#     assert X_past.shape == (94, ylags, MX+MY)
#     assert X_future.shape == (94, tlags, MX+MY)
#     assert y_future.shape == (94, lags, MY)
#
#     # X[0,0,x] is y_future[0] - 2 -> 3
#     assert X_future[0,0,0] == 3


def test_8t():
    xlags = XLAGS           # changed
    ylags = YLAGS
    tlags = TLAGS

    tt = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        transpose=False,
        flatten=False,
        concat=True,    # changed
        encoder=None,
        decoder=-1
    )

    X_past_future, y_future = tt.fit_transform(y=yt, X=Xt)

    assert isinstance(X_past_future, tuple) and len(X_past_future) == 2
    X_past, X_future = X_past_future

    assert X_past.shape == (94, ylags, MX+MY)
    assert X_future.shape == (94, tlags, MX+MY)
    assert y_future.shape == (94, tlags, MY)

    # X[0,0,x] is y_future[0] - 2 -> 3
    assert X_future[0,0,0] == 4


def test_8p():
    xlags = XLAGS           # changed
    ylags = YLAGS
    tlags = TLAGS

    tt = LagsTrainTransform(
        xlags=xlags, ylags=ylags, tlags=tlags,
        transpose=False,
        flatten=False,
        concat=True,    # changed
        encoder=None,
        decoder=-1
    )

    pt = tt.predict_transform()

    y_pred = pt.fit(y=yt, X=Xt).transform(fh=fh,X=Xp)
    X_pred_decoder = pt.step(0)

    assert isinstance(X_pred_decoder, tuple) and len(X_pred_decoder) == 2

    X_pred, X_decoder = X_pred_decoder

    assert y_pred.shape == (fh, MY)
    assert X_pred.shape == (1, ylags, MX+MY)
    assert X_decoder.shape == (1, tlags, MX+MY)

    assert X_decoder[0, 0, 0] == 100    # last y_past
    assert X_decoder[0, 1, 0] == 0      # first y_future NOT EXISTENT yet
    assert X_decoder[0, 0, 1] == 100.1  # last X_past
    assert X_decoder[0, 1, 1] == 101.1  # first X_future

