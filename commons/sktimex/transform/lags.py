import numpy as np

from ._base import ARRAY_OR_DF, ModelTrainTransform, ModelPredictTransform
from ..utils import is_instance, NoneType, RangeType
from ._lags import lmax


def _transpose(t):
    return None if t is None else t.swapaxes(1, 2)


def _flatten(t, n):
    return None if t is None else t.reshape(n, -1)


def _concat(*tlist):
    # remove None values
    tlist = [t for t in tlist if t is not None]
    if len(tlist) == 0:
        return None
    if len(tlist) == 1:
        return tlist[0]
    else:
        return np.concatenate(tlist, axis=-1)


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
    It receives in input 2 matrices (2D tensors), 'y' and 'X', and creates 2 object ('Xt', 'yt') that can
    be used with 'model.fit(Xt, yt)'. The objects can be:

        - 2D or 3D tensors
        - tuples of 2D or 3D tensors

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
         'all':     (n, len(ylags)*ny + len(xlags)*nx + len(tlags)*nx)

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
    """

    def __init__(self,
                 xlags, ylags, tlags,
                 transpose=False, flatten=False, concat=True,
                 encoder=None, decoder=None):
        """
        Initialize the transformer
        :param xlags: lags used for X data. It can be None, an int, the empty list or a list of integers
        :param ylags: lags used for y data. It must be specified
        :param tlags: last used for predicted data. It must be specified
        :param transpose: if to transpose the 3D tensors
        :param flatten: if to flatten the 3D tensors into 2D
        :param concat:  if to concat y and X used for input, into a single tensor
        :param encoder: if to create the output tensor for the encoder. In this case its value is the time offset (>=0)
        :param decoder: if to create the input  tensor for the decoder. In this case its value is the time offset (<=0)
        """

        assert is_instance(xlags, (NoneType, int, list[int], RangeType)), f"Invalid 'xlags' value: {xlags}"
        assert is_instance(ylags, (int, list[int], RangeType)), f"Invalid 'ylags' value: {ylags}"
        assert is_instance(tlags, (int, list[int], RangeType)), f"Invalid 'tlags' value: {tlags}"
        assert is_instance(encoder, (NoneType, int, str)), f"Invalid 'encoder' value: {encoder}"
        assert is_instance(decoder, (NoneType, int, str)), f"Invalid 'decoder' value: {decoder}"

        super().__init__(xlags=xlags, ylags=ylags, tlags=tlags)

        self.flatten = flatten
        self.concat = concat
        self.transpose = transpose
        self.encoder = encoder
        self.decoder = decoder
    # end

    def transform(self, y: ARRAY_OR_DF = None, X: ARRAY_OR_DF = None, fh=None) -> tuple:
        X, y = self._check_Xy(X, y, fh)

        all_data = self._prepare_data(X, y)
        X_fit, y_predict = self._compose_data(all_data)
        return X_fit, y_predict
    # end

    def _prepare_data(self, X, y):
        xlags = self.xlags if X is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)
        s = max(lmax(xlags), max(ylags))  # ylags MUST contain at minimum 1 value
        t = max(tlags) + 1  # tlags MUST contain at minimum 1 value ([0])
        r = s + t

        mx = X.shape[1] if sx > 0 else 0
        my = y.shape[1]
        n = y.shape[0] - r

        # prepare y_past
        y_past = np.zeros((n, sy, my), dtype=y.dtype)
        for i in range(n):
            for c, j in enumerate(reversed(ylags)):
                y_past[i, c, :] = y[s + i - j]

        # prepare X_past
        X_past = np.zeros((n, sx, mx), dtype=y.dtype) if sx > 0 else None
        for i in range(n):
            for c, j in enumerate(reversed(xlags)):
                X_past[i, c, :] = X[s + i - j]

        # prepare y_future
        y_future = np.zeros((n, st, my), dtype=y.dtype)
        for i in range(n):
            for c, j in enumerate(tlags):
                y_future[i, c, :] = y[s + i + j]

        # prepare X_future
        X_future = np.zeros((n, st, mx), dtype=y.dtype) if sx > 0 else None
        ulags = tlags if sx > 0 else []
        for i in range(n):
            for c, j in enumerate(ulags):
                X_future[i, c, :] = X[s + i + j]

        # prepare y_encoder
        if self.encoder is None:
            y_encoder = None
        elif isinstance(self.encoder, int):
            shift = self.encoder
            y_encoder = np.zeros((n, sy, my), dtype=y.dtype)
            for i in range(n):
                for c, j in enumerate(reversed(ylags)):
                    y_encoder[i, c, :] = y[s + i - j + shift]
        else:
            raise ValueError(f"Unsupported 'encoder' parameter: {self.encoder}")

        # prepare X_decoder, y_decoder
        # Note: if decoder is 0, only X_decoder is valid
        if self.decoder is None:
            X_decoder = None
            y_decoder = None
        elif self.decoder == "past":
            shift = -st
            y_decoder = np.zeros((n, st, my), dtype=y.dtype)
            for i in range(n):
                for c, j in enumerate(tlags):
                    y_decoder[i, c, :] = y[s + i + j + shift]
            X_decoder = np.zeros((n, st, mx), dtype=y.dtype) if sx > 0 else None
            ulags = tlags if sx > 0 else []
            for i in range(n):
                for c, j in enumerate(ulags):
                    X_decoder[i, c, :] = X[s + i + j + shift]
        elif self.decoder == 0:
            y_decoder = None
            X_decoder = X_future
        elif isinstance(self.decoder, int):
            assert self.decoder < 0, "Unsupported 'decoder' shift greater than 0"
            shift = self.decoder
            y_decoder = np.zeros((n, st, my), dtype=y.dtype)
            for i in range(n):
                for c, j in enumerate(tlags):
                    y_decoder[i, c, :] = y[s + i + j + shift]
            X_decoder = np.zeros((n, st, mx), dtype=y.dtype) if sx > 0 else None
            ulags = tlags if sx > 0 else []
            for i in range(n):
                for c, j in enumerate(ulags):
                    X_decoder[i, c, :] = X[s + i + j + shift]
            pass
        else:
            raise ValueError(f"Unsupported 'decoder' parameter: {self.decoder}")

        return (X_past, y_past), (X_future, y_future), y_encoder, (X_decoder, y_decoder)
    # end

    def _compose_data(self, all_data: tuple):

        Xy_past, Xy_future, y_encoder, Xy_decoder = all_data
        X_past, y_past = Xy_past
        X_future, y_future = Xy_future
        X_decoder, y_decoder = Xy_decoder
        n = len(y_past)

        # check xlags, ylags compatibility
        if not self.flatten:
            if self.concat in [True, "xy", "xyonly"]:
                xlags = self.xlags if X_past is not None else []
                ylags = self.ylags
                if xlags != [] and xlags != ylags:
                    raise ValueError(f"Incompatible 'xlags' vs 'ylags' when 'flatten' is False: {xlags}, {ylags}")
            elif self.concat in ['x', 'xonly']:
                raise ValueError(f"Incompatible 'concat' mode when 'flatten' is False: {self.concat}")

        if self.transpose:
            X_past    = _transpose(X_past)
            y_past    = _transpose(y_past)
            X_future  = _transpose(X_future)
            y_future  = _transpose(y_future)
            y_encoder = _transpose(y_encoder)
            X_decoder = _transpose(X_decoder)
            y_decoder = _transpose(y_decoder)

        if self.flatten:
            X_past    = _flatten(X_past, n)
            y_past    = _flatten(y_past, n)
            X_future  = _flatten(X_future, n)
            y_future  = _flatten(y_future, n)
            y_encoder = _flatten(y_encoder, n)
            X_decoder = _flatten(X_decoder, n)
            y_decoder = _flatten(y_decoder, n)

        if self.concat in [None, False]:
            X_fit = y_past
            X_fit_decoder = y_decoder
        elif self.concat in [True, 'xy', 'xyonly']:
            X_fit = _concat(y_past, X_past)
            X_fit_decoder = _concat(y_decoder, X_decoder)
        elif self.concat in ["x", "xonly"]:
            X_fit = _concat(X_past, X_future)
            X_fit_decoder = X_decoder
        elif self.concat in ["all", all]:
            X_fit = _concat(y_past, X_past, X_future)
            X_fit_decoder = _concat(y_decoder, X_decoder)
        else:
            raise ValueError(f"Unsupported 'concat' mode: {self.concat}")

        if self.encoder is not None:
            y_future = (y_encoder, y_future)

        if self.decoder is not None:
            X_fit = (X_fit, X_fit_decoder)

        return X_fit, y_future
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
            transpose=self.transpose,
            flatten=self.flatten,
            concat=self.concat,
            encoder=self.encoder,
            decoder=self.decoder
        )
    # end
# end


class LagsPredictTransform(ModelPredictTransform):

    def __init__(self, xlags, ylags, tlags,
                 transpose=False, flatten=False, concat=True,
                 encoder=None, decoder=None):

        assert is_instance(xlags, (NoneType, int, list[int], RangeType)), f"Invalid 'xlags' value: {xlags}"
        assert is_instance(ylags, (int, list[int], RangeType)), f"Invalid 'ylags' value: {ylags}"
        assert is_instance(tlags, (int, list[int], RangeType)), f"Invalid 'tlags' value: {tlags}"
        assert is_instance(encoder, (NoneType, int)), f"Invalid 'encoder' value: {encoder}"
        assert is_instance(decoder, (NoneType, int)), f"Invalid 'decoder' value: {encoder}"

        super().__init__(xlags=xlags, ylags=ylags, tlags=tlags)

        self.flatten = flatten
        self.concat = concat
        self.transpose = transpose
        self.encoder = encoder
        self.decoder = decoder

        self.X_past = None
        self.y_past = None
        self.yp = None
        self.y_encoder = None
        self.X_decoder = None
        self.y_decoder = None
    # end

    def transform(self, fh: int = 0, X: ARRAY_OR_DF = None, y=None):
        X, fh = self._check_Xfh(X, fh, y)

        Xh = self.Xh
        yh = self.yh
        self.Xp = X
        self.fh = fh

        y_pred = self._prepare_data(X, fh)
        # return self.to_pandas(y_pred)
        return y_pred

    def _prepare_data(self, X, fh):
        # saved with 'fit'
        Xh = self.Xh
        yh = self.yh

        xlags = self.xlags if Xh is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

        mx = Xh.shape[1] if sx > 0 else 0
        my = yh.shape[1]

        X_past = np.zeros((1, sx, mx), dtype=yh.dtype) if sx > 0 else None
        y_past = np.zeros((1, sy, my), dtype=yh.dtype)
        y_future = np.zeros((fh, my), dtype=yh.dtype)

        if self.encoder is None:
            y_encoder = None
        else:
            y_encoder = np.zeros((1, sy, my), dtype=yh.dtype)

        if self.decoder is None:
            X_decoder = None
            y_decoder = None
        elif self.decoder == 0:
            X_decoder = np.zeros((1, st, mx), dtype=yh.dtype) if sx > 0 else None
            y_decoder = None
        else:
            X_decoder = np.zeros((1, st, mx), dtype=yh.dtype) if sx > 0 else None
            y_decoder = np.zeros((1, st, my), dtype=yh.dtype)

        self.X_past = X_past
        self.y_past = y_past
        self.yp = y_future

        self.y_encoder = y_encoder
        self.X_decoder = X_decoder
        self.y_decoder = y_decoder

        return y_future
    # end

    def _xat(self, i):
        return self.Xh[i] if i < 0 else self.Xp[i]

    def _yat(self, i):
        return self.yh[i, 0] if i < 0 else self.yp[i, 0]

    def step(self, i, t=None) -> tuple:  # Xt, yt, Xf
        self._fill_data(i, t)
        Xpred = self._compose_data()
        return Xpred
    # end

    def _fill_data(self, i, t):
        xat = self._xat
        yat = self._yat

        xlags = self.xlags if self.Xh is not None else []
        ylags = self.ylags
        tlags = self.tlags

        sx = len(xlags)
        sy = len(ylags)
        st = len(tlags)

        X_past = self.X_past
        y_past = self.y_past
        # y_encoder = self.y_decoder
        X_decoder = self.X_decoder
        y_decoder = self.y_decoder

        # prepare X_past, y_past
        for c, j in enumerate(reversed(ylags)):
            y_past[0, c, :] = yat(i - j)
        for c, j in enumerate(reversed(xlags)):
            X_past[0, c, :] = xat(i - j)

        # prepare X_decoder, y_decoder
        if self.decoder is None:
            pass
        elif self.decoder == 'past':
            shift = -st
            for c, j in enumerate(tlags):
                y_decoder[0, c, :] = yat(i + j + shift)
            ulags = tlags if sx > 0 else []
            for c, j in enumerate(ulags):
                X_decoder[0, c, :] = xat(i + j + shift)
        elif self.decoder == 0:
            ulags = tlags if sx > 0 else []
            for c, j in enumerate(ulags):
                X_decoder[0, c, :] = xat(i + j)
        elif isinstance(self.decoder, int):
            assert self.decoder < 0, "Unsupported 'decoder' shift greater than 0"
            # shift = self.decoder - max(tlags)
            shift = self.decoder
            for c, j in enumerate(tlags):
                y_decoder[0, c, :] = yat(i + j + shift)
            ulags = tlags if sx > 0 else []
            for c, j in enumerate(ulags):
                X_decoder[0, c, :] = xat(i + j + shift)

        return
    # end

    def _compose_data(self):
        X_past = self.X_past
        y_past = self.y_past
        # y_encoder = self.y_decoder
        X_decoder = self.X_decoder
        y_decoder = self.y_decoder

        n = len(y_past)
        if self.transpose:
            X_past = _transpose(X_past)
            y_past = _transpose(y_past)
            # y_encoder = _transpose(y_encoder)
            X_decoder = _transpose(X_decoder)
            y_decoder = _transpose(y_decoder)

        if self.flatten:
            X_past = _flatten(X_past, n)
            y_past = _flatten(y_past, n)
            # y_encoder = _flatten(y_encoder, n)
            X_decoder = _flatten(X_decoder, n)
            y_decoder = _flatten(y_decoder, n)

        if self.concat in [None, False]:
            X_predict = y_past
            X_decoder = y_decoder
        elif self.concat in [True, 'xy', 'xyonly']:
            X_predict = _concat(y_past, X_past)
            X_decoder = _concat(y_decoder, X_decoder)
        elif self.concat in ["x", "xonly"]:
            X_predict = _concat(X_past)
            X_decoder = X_decoder
        elif self.concat in ["all", all]:
            X_predict = _concat(y_past, X_past)
            X_decoder = _concat(y_decoder, X_decoder)
        else:
            raise ValueError(f"Unsupported 'concat' mode: {self.concat}")

        if self.decoder is not None:
            X_predict = (X_predict, X_decoder)

        return X_predict
    # end

    def update(self, i, y_pred, t=None):
        return super().update(i, y_pred, t)
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
