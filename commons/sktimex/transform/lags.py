import numpy as np

from .base import ARRAY_OR_DF, ModelTrainTransform, ModelPredictTransform
from ..lags import lmax


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_Xy(Xp, yp, Xf, yf, *, transpose, flatten, concat, xylags):
    """

    :param Xp: X past (optional)
    :param yp: y past
    :param Xf: X future (optional)
    :param yf: y future
    :param transpose: if to transpose the last 2 axes
    :param flatten: if to flatten
    :param concat: if to concat Xp with yp and/or Xf
    :param xylags: if xlags is compatible with ylags (xlags == ylags or xlags == [])
    :return: the tuple Xp', yp', Xf', yf' after all transformations
    """
    # It receives in input 3D tensors with dimensions:
    #
    #   (N, seq_len, data_len)
    #
    # 'transpose' has effect on 'flatten'
    # If NOT transposed, after flattening the data structure will be:
    #
    #       x1[t+1],x1[t+2] ... x2[t+1],x2[t+2],...
    #
    # If TRANSPOSED, after flattening the data structure will be:
    #
    #       x1[t+1],x2[t+1] ... x1[t+2],x2[t+2],...
    #
    #
    if transpose:
        Xp = np.swapaxes(Xp, 1, 2) if Xp is not None else None
        yp = np.swapaxes(yp, 1, 2)
        Xf = np.swapaxes(Xf, 1, 2) if Xf is not None else None
        yf = np.swapaxes(yf, 1, 2) if yf is not None else None

    if flatten:
        n = len(yp)
        Xp = Xp.reshape(n, -1) if Xp is not None else None
        yp = yp.reshape(n, -1)
        Xf = Xf.reshape(n, -1) if Xf is not None else None
        yf = yf.reshape(n, -1) if yf is not None else None

    if Xp is None:
        if concat:
            Xp = yp
        else:
        # return Xp, yp, Xf, yf
            pass

    elif concat == False:
        # return Xp, yp, Xf, yf
        pass

    elif flatten:
        if concat == True:
            # Xp = np.concatenate([Xp, yp], axis=1)
            Xp = np.concatenate([yp, Xp], axis=1)
        elif concat == 'xonly':
            Xp = np.concatenate([Xp, Xf], axis=1)
        elif concat == 'all' or concat == all:
            # Xp = np.concatenate([Xp, yp, Xf], axis=1)
            Xp = np.concatenate([yp, Xp, Xf], axis=1)
        else:
            raise ValueError(f"Invalid concat mode in 2D: concat={concat}")
    else:
        if concat == True and xylags:
            # Xp = np.concatenate([Xp, yp], axis=2)
            Xp = np.concatenate([yp, Xp], axis=2)
        else:
            raise ValueError(f"Invalid concat mode in 3D: concat={concat}, xylags={xylags}")

    return Xp, yp, Xf, yf
# end


# ---------------------------------------------------------------------------
# LagsTrainTransform
# LagsPredictTransform
# ---------------------------------------------------------------------------
# (X, y, xslot, yslots) -> Xt, yt
#
# back_step
#   y[-1]             -> y[0]
#   y[-1],X[-1]       -> y[0]
#   y[-1],X[-1],X[0]  -> y[0]
#

class LagsTrainTransform(ModelTrainTransform):
    """
    Transformer used to prepare the tensors used in the model's training.
    It receives in input 2 matrices (2D tensors), 'y' and 'X', and creates 4 tensors based on
    'xlags', 'ylags' and 'tlags'

    1) y_past, based on ylags
    2) X_past, based on xlags, if X and xlags are specified (X is not None, xlags is not None or the empty list)
    3) y_future, based on tlags
    4) X_future, based on tlags, if X and xlags are specified

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

    The parameter 'flatten' is used to return 2D tensors instead than 3D tensors, with dimensions:

        y_past:     (n, len(ylags) * ny)
        X_past:     (n, len(xlags) * nx)
        y_future:   (n, len(tlags) * ny)
        X_future:   (n, len(tlags) * nx)

    The parameter 'concat' is used to specify how to compose X_past'. Possible values are:

        False:      X_past as is
        True:       concatenate [y_past, X_past]
        'xonly':    concatenate [X_past, X_future]
        'all':      concatenate [y_past, X_past, X_future]

    Note: 'y_past' is located at the tensor's head and if 'X' is not available, 'X_past' is equal to 'y_past'

    The concatenation depends on 'flatten': if 'flatten' is true, the 2D tensors are concatenated horizontally,
    generating 'X_past' with dimensions:

         True:      (n, len(ylags)*ny + len(xlags)*nx)
         'xonly':   (n, len(xlags)*nx + len(tlags)*nx)
         'all':     (n, llen(ylags)*ny + en(xlags)*nx + len(tlags)*nx)

    If 'flatten' is false, the concatenation is possible only if 'xlags == ylags' or 'xlags == []' (the empty list).
    Because 'tlags' will be different than 'ylags' (and 'xlags'), the concatenation modes 'xonly' and 'all'
    will be invalid. In this case the 3D tensors are concatenated on the 3^rd axis:

        True:       (n, len(xlags), ny+nx)      ('y_past' in front)
        'xonly':    invalid
        'all':      invalid

    The parameter 'transpose' is valid only if 'flatten' is false. It is used to create tensors compatible with
    CNN layers using tensors with dimensions (n, channels, channel_size). In this case, it swap the axes
    'seq_len' with 'data_size', creating tensors with dimensions

        (n_elements, data_size, seq_len)

    :param slots:
    :param xlags: past lags used for X
    :param ylags: past lags used for y
    :param tlags: future lags used for y (and X, if required)
    :param flatten: if to return a 2D tensor (n, seq_len*data_size) or a 3D tensor (n, seq_len, data_size)
    :param concat:  if to concatenate X_past, y_past, X_future (see the previous description)
    :param transpose: if to transpose axis 1 with axis 2 (valid only if flatten is false)
    """

    def __init__(self, slots=None,
                 xlags=None, ylags=None, tlags=(0,),
                 flatten=False, concat=True, transpose=False):
        """
        Initialize the transformer

        :param slots: (internal usage)
        :param xlags: past lags used for X
        :param ylags: past lags used for y
        :param tlags: future lags used for y (and X, if required)
        :param flatten: if to return a 2D tensor (n, seq_len*data_size) or a 3D tensor (n, seq_len, data_size)
        :param concat:  if to concatenate y_past, X_past, X_future (see the previous description)
        :param transpose: if to transpose axis 1 with axis 2 (valid only if flatten is false)
        """
        if ylags is not None:
            slots = [xlags, ylags]
        super().__init__(slots=slots, tlags=tlags)
        self.flatten = flatten
        self.concat = concat
        self.transpose = transpose
    # end

    def transform(self, y: ARRAY_OR_DF = None, X: ARRAY_OR_DF=None, fh=None) -> tuple:
        """
        Compose y and X to create the 4 tensors used to train the time series models.
        It returns 4 tensors: X_past, y_past, X_future, y_future.
        If necessary, X_past con be the composition of y_past, X_past, X_future

        :param y: 2D targets
        :param X: 2D input features
        :param fh: not used, but present for interface compatibility
        :return: the tensors X_past, y_past, X_future, y_future
        """
        X, y = self._check_Xy(X, y, fh)

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        # s = len(self.slots)
        s = max(lmax(xlags), lmax(ylags))   # ylags MUST contain at minimum 1 value
        t = max(tlags) + 1                  # tlags MUST contain at minimum 1 value ([0])
        r = s + t

        mx = X.shape[1] if sx > 0 else 0
        my = y.shape[1]
        n = y.shape[0] - r

        Xp = np.zeros((n, sx, mx), dtype=y.dtype) if sx > 0 else None
        yp = np.zeros((n, sy, my), dtype=y.dtype)
        Xf = np.zeros((n, st, mx), dtype=y.dtype) if sx > 0 else None
        yf = np.zeros((n, st, my), dtype=y.dtype)

        for i in range(n):
            for c, j in enumerate(reversed(ylags)):
                yp[i, c, :] = y[s + i - j]
            for c, j in enumerate(reversed(xlags)):
                Xp[i, c, :] = X[s + i - j]
        # end

        for i in range(n):
            for c, j in enumerate(tlags):
                yf[i, c, :] = y[s + i + j]
            for c, j in enumerate(tlags):
                if Xf is not None:
                    Xf[i, c, :] = X[s + i + j]
        # end

        Xp, yp, Xf, yf = normalize_Xy(Xp, yp, Xf, yf,
                                      flatten=self.flatten, concat=self.concat, transpose=self.transpose,
                                      xylags=xlags == [] or xlags==ylags)

        # X_past, y_past, X_future, y_future
        return Xp, yp, Xf, yf
    # end

    def predict_transform(self):
        """
        This method return the correspondent 'LagsPredictTransform' having the same configuration
        parameters of this transformer
        """
        return LagsPredictTransform(
            xlags=self.xlags,
            ylags=self.ylags,
            tlags=self.tlags,
            flatten=self.flatten,
            concat=self.concat,
            transpose=self.transpose
        )
# end


class LagsPredictTransform(ModelPredictTransform):
    """
    Transformer used to prepare the tensors used in the model's prediction.
    It creates a 2D tensor used to save the predicted value and to use them in recursive way to
    generate all values in the prediction horizon based on the configured forecasting horizon.

    It is used in the following way:

        pt = LagsPredictTransform(xlags=.., ylags=.., tlags=.., ...)

        # create the 2D tensors used to save the predicted values
        y_predict = pt.fit(y_past, X_past).transform(fh=FH_LEN, X_predict)

        i = 0
        while i < FH_LEN:
            # prepare the (1, ..) tensors to use with the model
            Xt, yt, Xf = pt.step(i)

            # call the model to predict the complete forecasting horizon (1 or more time steps)
            yp = model.predict(Xt, yt, Xf)

            # update 'y_predict' collecting the predicted values from 'yp'
            # then advances the prediction window of the forecasting horizon time slots
            i = pt.update(i, yp)
        # end

    The parameters used in the constructor must be the same of the parameters used for
    'LagsTrainTransform', transformer used to prepare the data for the training.

    Note: 'Xt', 'yt', 'Xf' are tensors updated at each iteration. It is necessary to clone
          them if their values must be saved

    """

    def __init__(self, slots=None, xlags=None, ylags=None, tlags=(0,),
                 flatten=False, concat=True, transpose=False):
        """
        Initialize the transformer

        :param slots: (internal usage)
        :param xlags: past lags used for X
        :param ylags: past lags used for y
        :param tlags: future lags used for y (and X, if required)
        :param flatten: if to return a 2D tensor (n, seq_len*data_size) or a 3D tensor (n, seq_len, data_size)
        :param concat:  if to concatenate X_past, y_past, X_future (see the previous description)
        :param transpose: if to transpose axis 1 with axis 2 (valid only if flatten is false)
        """
        if ylags is not None:
            slots = [xlags, ylags]
        super().__init__(slots=slots, tlags=tlags)
        self.flatten = flatten
        self.concat = concat
        self.transpose = transpose
    # end

    def transform(self, fh: int = 0, X: ARRAY_OR_DF = None, y=None):
        fh, X = super().transform(fh, X, y)

        Xh = self.Xh
        yh = self.yh

        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        # s = len(self.slots)
        # t = lmax(tlags)
        # r = t + s

        mx = Xh.shape[1] if sx > 0 else 0
        my = yh.shape[1]

        Xt = np.zeros((1, sx, mx), dtype=yh.dtype) if mx > 0 else None
        yt = np.zeros((1, sy, my), dtype=yh.dtype)
        Xf = np.zeros((1, st, mx), dtype=yh.dtype) if mx > 0 else None
        yp = np.zeros((fh, my), dtype=yh.dtype)

        self.Xt = Xt    # (1, sx, mx)
        self.yt = yt    # (1, sy, my)
        self.Xf = Xf    # (1, st, mx)
        self.Xp = X     #
        self.yp = yp    # (fh, my)

        return self.to_pandas(yp)
    # end

    def _xat(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _yat(self, i):
        return self.yh[i, 0] if i < 0 else self.yp[i, 0]

    def step(self, i) -> tuple:     # Xt, yt, Xf
        xat = self._xat
        yat = self._yat

        xlags = self.xlags if self.Xh is not None else []
        ylags = self.ylags
        tlags = self.tlags if self.Xf is not None else []

        Xt = self.Xt
        yt = self.yt
        Xf = self.Xf

        # prepare Xt, yt, Xf
        for c, j in enumerate(reversed(ylags)):
            yt[0, c, :] = yat(i - j)
        for c, j in enumerate(reversed(xlags)):
            Xt[0, c, :] = xat(i - j)
        for c, j in enumerate(tlags):
            Xf[0, c, :] = xat(i + j)

        Xt, yt, Xf, _ = normalize_Xy(Xt, yt, Xf, None,
                                     flatten=self.flatten, concat=self.concat, transpose=self.transpose,
                                     xylags=xlags == [] or xlags==ylags)

        return Xt, yt, Xf

    def update(self, i, y_pred, t=None):
        return super().update(i, y_pred, t)
# end

