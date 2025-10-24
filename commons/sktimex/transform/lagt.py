from typing import Optional

import numpy as np

from pandasx import to_numpy
from ._base import TimeseriesTransform
from ._utils import _transpose, _flatten, _concat
from ._utils import lmax, t_step, t_start


# ---------------------------------------------------------------------------
# LagsTrainTransform
# LagsPredictTransform
# ---------------------------------------------------------------------------
# Extends tlags
#   lags    (0-start)
#                y  X
#       n       [n, n]
#       [n]     [n, 0]
#       [,m]    [0, m]
#       [n,m]   [n, m]
#
#       n, m integers
#       n -> [0...n-1]  (same for m)
#       instead than 'n' it is possible to specify a list of lags
#
#   tlags   (1-start, compatible with sktime)
#                y  X
#       n       [n, 0]
#       [n]     [n, 0]
#       [n,m]   [n, m]
#
#       n, m integers
#       n -> [1...n]  (same for m)
#       instead than 'n' it is possible to specify a list of lags
#

class LagsTrainTransform(TimeseriesTransform):
    """
    Transformer used to prepare the tensors used in the model's training.
    It receives in input 2 matrices (2D tensors), 'y' and 'X', and creates 2 objects ('Xt', 'yt') that can
    be used with 'model.fit(Xt, yt)' of a scikit-learn model. The objects can be:

        - 2D or 3D tensors
        - tuples of 2D or 3D tensors

    Lags used: 'xlags', 'ylags' and 'tlags'

    1) y_past, based on ylags
    2) X_past, based on xlags, if X and xlags are specified.
       If X is None or xlags iis the empty list, the result is None
    3) y_future, based on tlags
    4) X_future, based on tlags, if X and xlags are specified.

    having dimensions

        (n_elements, seq_len, data_size)

    All tensors are aligned such that the next timeslot of 'y_past' corresponds to the first timeslot of 'y_future'.
    The same for 'X_past' vs 'X_future'

    Let 'y' with dimensions (N, ny) and 'X' with dimensions (N, nx). The native dimensions of the tensors will be:

        y_past:     (n, len(ylags), ny)
        X_past:     (n, len(xlags), nx)
        y_future:   (n, len(tlags), ny)
        X_future:   (n, len(tlags), nx)

    where n < N, is computed considering 'xlags', 'ylags' and 'tlags' (the formula is a little complex because
    it doesn't depend on lags's sized but on the highest timeslot index used).

    The parameter 'flatten' is used to return 2D tensors instead of 3D tensors, with dimensions:

        y_past:     (n, len(ylags) * ny)
        X_past:     (n, len(xlags) * nx)
        y_future:   (n, len(tlags) * ny)
        X_future:   (n, len(tlags) * nx)

    The parameter 'concat' is used to specify how to compose X_past'. Possible values are:

        False:      X_past as is
        True:       concatenate [y_past, X_past]
        'xonly':    concatenate [X_past, X_future]
        'all':      concatenate [y_past, X_past, X_future]

    Note: 'y_past' is located at the tensor's head, and if 'X' is not available, 'X_past' is equal to 'y_past'

    The concatenation depends on 'flatten': if 'flatten' is true, the 2D tensors are concatenated horizontally,
    generating 'X_past' with dimensions:

         True:      (n, len(ylags)*ny + len(xlags)*nx)
         'xonly':   (n, len(xlags)*nx + len(tlags)*nx)
         'all':     (n, len(ylags)*ny + len(xlags)*nx + len(tlags)*nx)

    If 'flatten' is false, the concatenation is possible only if 'xlags == ylags' or 'xlags == []' (the empty list).
    Because 'tlags' will be different from 'ylags' (and 'xlags'), the concatenation modes 'xonly' and 'all'
    will be invalid. In this case the 3D tensors are concatenated on the 3^rd axis:

        True:       (n, len(xlags), ny+nx)      ('y_past' in front)
        'xonly':    invalid
        'all':      invalid

    The parameter 'transpose' is valid only if 'flatten' is false. It is used to create tensors compatible with
    CNN layers using tensors with dimensions (n, channels, channel_size). In this case, it swaps the axes
    'seq_len' with 'data_size', creating tensors with dimensions

        (n_elements, data_size, seq_len)
    """

    def __init__(
            self, *,
            xlags, ylags, tlags, ulags,
            flatten=True, concat=True, transpose=False,
        ):
        """
        Initialize the transformer
        :param xlags: lags used for X data. It can be None, an int, the empty list or a list of integers
        :param ylags: lags used for y data. It must be specified
        :param tlags: last used for predicted data. It must be specified
        :param transpose: if to transpose the 3D tensors
        :param flatten: if to flatten the 3D tensors into 2D
        :param concat:  if to concat y and X used for input, into a single tensor
        """
        super().__init__(xlags, ylags, tlags, ulags)

        self.flatten = flatten
        self.concat = concat
        self.transpose = transpose

        # -------------------------------------------------------------------

        self._X = None
        self._y = None
        self._fh = None
    # end

    # ---------------------------------------------------------------------------
    # transform
    # ---------------------------------------------------------------------------
    
    def fit(self, *, y, X=None):
        super().fit(y=y, X=X)

        X, y = self._check_Xy(X, y)

        self._X = X
        self._y = y

        return self
    # end

    def transform(self, *, fh=None, y=None, X=None) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._check_Xy(X, y, fh)

        if self.flatten and self.concat is True and not self.transpose:
            # efficient version
            Xt, yt = self._prepare_compose_data(y, X)
        else:
            all_data = self._prepare_data(y, X)
            Xt, yt = self._compose_data(all_data)

        return Xt, yt
    # end

    # ---------------------------------------------------------------------------
    # data preparation
    # ---------------------------------------------------------------------------

    def _prepare_compose_data(self, y, X):
        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags
        ulags = self.ulags if X is not None else []

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        su = len(ulags)

        s = max(lmax(xlags), lmax(ylags))
        t = max(lmax(tlags), lmax(ulags))
        r = s + t

        oy = lmax(ylags)
        ox = lmax(xlags)
        ou = s

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        n = y.shape[0] - r

        wx = sy * my + sx * mx + su * mx
        wy = st * my

        Xt = np.zeros((n, wx), dtype=y.dtype)
        yt = np.zeros((n, wy), dtype=y.dtype)

        # prepare Xt
        for i in range(n):
            k = 0
            for c, j in enumerate(reversed(ylags)):
                Xt[i, k:k+my] = y[oy + i - j]
                k += my

            for c, j in enumerate(reversed(xlags)):
                Xt[i, k:k+mx] = X[ox + i - j]
                k += mx

            for c, j in enumerate(ulags):
                Xt[i, k:k+mx] = X[ou + i + j]
                k += mx

        # prepare yt
        for i in range(n):
            k = 0
            for c, j in enumerate(tlags):
                yt[i, k:k+my] = y[ou + i + j]
                k += my

        return Xt, yt
    #end

    def _prepare_data(self, y, X):
        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags
        ulags = self.ulags if X is not None else []

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        su = len(ulags)

        s = max(lmax(xlags), lmax(ylags))   # ylags MUST contain at minimum 1 value
        t = max(lmax(tlags), lmax(ulags))   # tlags MUST contain at minimum 1 value ([1])
        r = s + t

        ox = lmax(xlags)
        oy = lmax(ylags)
        ou = s

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]
        n = y.shape[0] - r

        # prepare y_past
        y_past = np.zeros((n, sy, my), dtype=y.dtype)
        for i in range(n):
            for c, j in enumerate(reversed(ylags)):
                y_past[i, c, :] = y[oy + i - j]

        # prepare X_past
        X_past = np.zeros((n, sx, mx), dtype=y.dtype) if sx > 0 else None
        for i in range(n):
            for c, j in enumerate(reversed(xlags)):
                X_past[i, c, :] = X[ox + i - j]

        # prepare y_future
        y_future = np.zeros((n, st, my), dtype=y.dtype)
        for i in range(n):
            for c, j in enumerate(tlags):
                y_future[i, c, :] = y[ou + i + j]

        # prepare X_future
        X_future = np.zeros((n, su, mx), dtype=y.dtype) if su > 0 else None
        for i in range(n):
            for c, j in enumerate(ulags):
                X_future[i, c, :] = X[ou + i + j]

        return (X_past, y_past), (X_future, y_future),
    # end

    def _compose_data(self, all_data: tuple):

        Xy_past, Xy_future = all_data
        X_past, y_past = Xy_past
        X_future, y_future = Xy_future
        n = len(y_past)

        # check xlags, ylags compatibility
        if not self.flatten and self.concat in [True, "xy", "xyonly"]:
            xlags = self.xlags if X_past is not None else []
            ylags = self.ylags
            if xlags != [] and xlags != ylags:
                raise ValueError(f"Incompatible 'xlags' vs 'ylags' when 'flatten' is False: {xlags}, {ylags}")

        if self.transpose:
            X_past    = _transpose(X_past)
            y_past    = _transpose(y_past)
            X_future  = _transpose(X_future)
            y_future  = _transpose(y_future)

        if self.flatten:
            X_past    = _flatten(X_past, n)
            y_past    = _flatten(y_past, n)
            X_future  = _flatten(X_future, n)
            y_future  = _flatten(y_future, n)

        if self.concat in [None, False]:
            X_fit = y_past
        elif self.concat in ['xy', 'xyonly']:
            X_fit = _concat(y_past, X_past)
        elif self.concat in ["x", "xonly"]:
            X_fit = _concat(X_past, X_future)
        elif self.concat in [True, "all", all]:
            X_fit = _concat(y_past, X_past, X_future)
        else:
            raise ValueError(f"Unsupported 'concat' mode: {self.concat}")

        return X_fit, y_future
    # end

    # ---------------------------------------------------------------------------
    # predict_transform
    # ---------------------------------------------------------------------------

    def predict_transform(self) -> "LagsPredictTransform":
        """
        This method returns the correspondent 'LagsPredictTransform' having the same configuration
        parameters of this transformer
        """
        return LagsPredictTransform(
            xlags=self.xlags,
            ylags=self.ylags,
            tlags=self.tlags,
            ulags=self.ulags,
            flatten=self.flatten,
            concat=self.concat,
            transpose=self.transpose,
        )
    # end
# end


class LagsPredictTransform(TimeseriesTransform):

    def __init__(
            self, *,
            xlags, ylags, tlags, ulags,
            flatten=True, concat=True, transpose=False,
    ):
        super().__init__(xlags, ylags, tlags, ulags)

        self.flatten = flatten
        self.concat = concat
        self.transpose = transpose

        # -------------------------------------------------------------------

        # set by fit
        self._Xh = None  # X history
        self._yh = None  # y history

        # set by predict
        self._Xp = None  # X prediction
        self._fh = None  # forecasting horizon

        # if tlags starts
        self._tstart: int = t_start(self.tlags)
        self._tstep = t_step(self.tlags)

        # used for composition
        self._X_past = None
        self._y_past = None
        self._X_future = None
        self._yp = None         # _y_future prediction, used as cache
        self._X_flat = None
    # end

    def fit_transform(self, *, y=None, X=None, fh=None):
        raise ValueError("This method is not supported. Use `fit(yh,Xh).transform(fh, Xf)` instead")

    def fit(self, *, y, X=None):
        super().fit(y=y, X=X)

        X, y = self._check_Xy(X, y)

        self._Xh = X    # X history
        self._yh = y    # Y history

        return self
    # end

    def transform(self, *, fh: Optional[int] = None, X=None, y=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the datasets for the predictions

        :param fh: forecasting horizon. It can be None if X is defined, in this case 'fh=len(X)'
        :param X: optional X
        :param y: must be None
        """
        X, nfh = self._check_Xfh(X, fh, y)

        self._Xp = X    # X prediction
        self._fh = nfh  # forecasting horizon

        if self.flatten and self.concat is True and not self.transpose:
            Xt, yt = self._prepare_compose_data()
        else:
            Xt = None
            yt = self._prepare_data()

        return Xt, yt
    # end

    def _check_Xfh(self, X, fh, y):
        X, nfh = super()._check_Xfh(X, fh, y)
        # st = max(self.tlags)

        assert self._yh is not None, "The 'yh' attribute must be set before calling 'transform', using 'fit(yh, Xh)"
        assert X is None and self._Xh is None or X is not None and self._Xh is not None, "X and Xh must be both None or not None"

        # note: IF len(tlags) is longer than 1, fh MUST BE a multiple than tlags
        # assert (nfh % st) == 0, f"When len(fh) must be a multiple tna len(tlags): fh={nfh}, tlags={self.tlags}"

        return X, nfh
    # end

    # ---------------------------------------------------------------------------
    # data preparation
    # ---------------------------------------------------------------------------

    def _xat(self, i):
        # for compatibility with sktime, the first slot of the future has index 1
        return self._Xh[i - 1] if i <= 0 else self._Xp[i - 1]

    def _yat(self, i):
        # for compatibility with sktime, the first slot of the future has index 1
        return self._yh[i - 1, 0] if i <= 0 else self._yp[i - 1, 0]

    def _prepare_compose_data(self):
        xat = self._xat
        yat = self._yat

        X = self._Xh
        y = self._yh
        nfh = self._fh

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags
        ulags = self.ulags if X is not None else []

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        su = len(ulags)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]

        wx = sy * my + sx * mx + su * mx
        # wy = st * my

        Xt = np.zeros((nfh, wx), dtype=y.dtype)
        yt = np.zeros((nfh, my), dtype=y.dtype)

        self._X_flat = Xt
        self._yp = yt

        for i in range(nfh):
            k = 0
            for c, j in enumerate(reversed(ylags)):
                Xt[i, k:k+my] = yat(i - j)
                k += my

            for c, j in enumerate(reversed(xlags)):
                Xt[i, k:k+mx] = xat(i - j)
                k += mx

            for c, j in enumerate(ulags):
                Xt[i, k:k+mx] = xat(i + j)
                k += mx

        return Xt, yt
    # end

    def _prepare_data(self):
        X = self._Xh
        y = self._yh
        nfh = self._fh

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags
        ulags = self.ulags if X is not None else []

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        su = len(ulags)

        mx = X.shape[1] if X is not None else 0
        my = y.shape[1]

        wx = sy * my + sx * mx + su * mx
        # wy = st * my

        self._X_past = np.zeros((1, sx, mx), dtype=y.dtype) if sx > 0 else None
        self._y_past = np.zeros((1, sy, my), dtype=y.dtype)
        self._X_future = np.zeros((1, su, mx), dtype=y.dtype) if su > 0 else None
        self._yp = np.zeros((nfh, my), dtype=y.dtype)
        self._X_flat = np.zeros((1, wx), dtype=y.dtype)

        return self._X_flat, self._yp
    # end

    # ---------------------------------------------------------------------------
    # step
    # ---------------------------------------------------------------------------

    def step(self, i, t=None) -> np.ndarray:
        if self.flatten and self.concat is True and not self.transpose:
            # Xs = self._fill_and_compose_data(i, t)
            Xs = self._X_flat[i:i+1]
        else:
            self._fill_data(i, t)
            Xs = self._compose_data()
        return Xs
    # end

    def _fill_and_compose_data(self, i):
        xat = self._xat
        yat = self._yat

        xlags = self.xlags if self._Xh is not None else []
        ylags = self.ylags
        ulags = self.ulags if self._Xp is not None else []

        sx = len(xlags)
        su = len(ulags)

        mx = self._Xh.shape[1] if self._Xh is not None else 0
        my = self._yh.shape[1]

        Xt = self._X_flat

        k = 0
        for c, j in enumerate(reversed(ylags)):
            Xt[0, k:k+my] = yat(i - j)
            k += my
        for c, j in enumerate(reversed(xlags)):
            Xt[0, k:k+mx] = xat(i - j)
            k += mx
        for c, j in enumerate(ulags):
            Xt[0, k:k+mx] = xat(i + j)
            k += mx

        return Xt
    # end

    def _fill_data(self, i, t):
        xat = self._xat
        yat = self._yat

        xlags = self.xlags if self._Xh is not None else []
        ylags = self.ylags if self._yh is not None else []

        # sx = len(xlags)
        # sy = len(ylags)
        # st = len(tlags)

        X_past = self._X_past
        y_past = self._y_past

        # prepare X_past, y_past
        for c, j in enumerate(reversed(ylags)):
            y_past[0, c, :] = yat(i - j)
        for c, j in enumerate(reversed(xlags)):
            X_past[0, c, :] = xat(i - j)

        return
    # end

    def _compose_data(self):
        X_past = self._X_past
        y_past = self._y_past

        n = len(y_past)
        if self.transpose:
            X_past = _transpose(X_past)
            y_past = _transpose(y_past)

        if self.flatten:
            X_past = _flatten(X_past, n)
            y_past = _flatten(y_past, n)

        if self.concat in [None, False]:
            X_predict = y_past
        elif self.concat in [True, 'xy', 'xyonly']:
            X_predict = _concat(y_past, X_past)
        elif self.concat in ["x", "xonly"]:
            X_predict = _concat(X_past)
        elif self.concat in ["all", all]:
            X_predict = _concat(y_past, X_past)
        else:
            raise ValueError(f"Unsupported 'concat' mode: {self.concat}")

        return X_predict
    # end

    # ---------------------------------------------------------------------------
    # update
    # ---------------------------------------------------------------------------

    def update(self, i, y_pred, t=None) -> int:
        i_next = self._update_yp(i, y_pred, t)

        if self.flatten and self.concat is True and not self.transpose:
            self._update_x_flat(i+1)

        return i_next
    # end

    def _update_x_flat(self, i):
        if i >= self._fh:
            return

        yat = self._yat
        ylags = self.ylags
        tlags = self.tlags
        st = len(tlags)
        my = self._yh.shape[1]

        Xt = self._X_flat

        for t in range(st):
            try:
                k = 0
                for c, j in enumerate(reversed(ylags)):
                    Xt[i+t, k:k + my] = yat(i - j + t)
                    k += my
            except:
                pass
        pass
    # end

    def _update_yp(self, i, y_pred, t=None) -> int:
        # Note:
        #   the parameter 't' is used to override tlags
        #   'tlags' is at minimum [1] (the next slot after cutoff)
        #
        # Extension:
        #   it is possible to use tlags=[-3,-2,-1,0,1]
        #   in this case, it is necessary to start with the position '4'!
        #   and advance 'i' ONLY of 2 slots.
        #   Really usable slots: [0, 1]
        assert isinstance(i, (int, np.int32)), "The argument 'i' must be the location update (an integer)"

        tlags = self.tlags if t is None else [t]
        tstart = self._tstart if t is None else 0

        st = len(tlags)  # length of tlags
        mt = max(tlags)  # max tlags index
        nfh = len(self._yp)  # length of fh
        my = self._yp.shape[1]  # predicted data size |y[i]|

        # convert y_pred as a 3D tensor
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((1, -1, my))
        elif len(y_pred.shape) == 2:
            y_pred = y_pred.reshape((1, -1, my))
        assert len(y_pred.shape) == 3

        # note tlags starts with 1
        for j in range(tstart, st):
            k = i + tlags[j] - 1
            if k < nfh:
                try:
                    self._yp[k] = y_pred[0, j]
                except IndexError:
                    pass

        # return i + mt
        return i + self._tstep
    # end

    # ---------------------------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------------------------

    def predict_steps(self, yf):
        # Note: step
        n = len(yf)
        Xt = self.step(0)
        xshape = (n,) + Xt.shape[1:]

        X_future = np.zeros(xshape, dtype=self._yh.dtype)
        y_future = self._yp

        i = 0
        while i < n:
            Xt = self.step(i)
            Xt = to_numpy(Xt)

            X_future[i, :] = Xt[0, :]

            # call the model
            yt = self._call_model(i, yf)
            yt = to_numpy(yt, matrix=True)

            i = self.update(i, yt)
            pass
        # end

        return X_future, y_future
    # end

    def _call_model(self, i, yf) -> np.ndarray:
        tlags = self.tlags
        yt = np.zeros((len(tlags),), dtype=self._yh.dtype)
        for c, j in enumerate(tlags):
            try:
                yt[c] = yf[i + j - 1]
            except IndexError:
                pass
        return yt
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
