from typing import Union, Optional
import numpy as np

__all__ = ['LagSlots', 'LagResolver', 'resolve_lag']

from numpy import ndarray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAG_INPUT = 'input'
LAG_TARGET = 'target'
LAG_DAY = 'day'
PERIOD_TYPE = 'period_type'
LAG_LENGTH = 'length'
LAG_CURRENT = 'current'

LAG_FACTORS: dict[str, int] = {
    '': 0,
    'second': 1,
    'minute': 60,
    'hour': 3600,
    'day': 3600 * 24,
    'week': 3600 * 24 * 7,
    # '4weeks': 1*60*60*24*7*4,
    'month': 3600 * 24 * 30,
    # '3months': 3600*24*30*3,
    # '4months': 3600*24*30*4,
    'quarter': 3600 * 24 * 91,  # 91 = 365//4 is a little better than 90 = 30*3
    'year': 3600 * 24 * 365,
}

LAG_TYPES: list[str] = list(LAG_FACTORS)


# ---------------------------------------------------------------------------
# LagSlots
# ---------------------------------------------------------------------------

def _max(l: list[int]) -> int:
    return 0 if len(l) == 0 else max(l)

class LagSlots:
    def __init__(self, islots: set[int], tslots: set[int]):
        self._islots: list[int] = sorted(islots)
        self._tslots: list[int] = sorted(tslots)

        # self._len = 0 if len(tslots) == 0 else max(max(islots), max(tslots)) + 1
        self._len = 0 if len(tslots) == 0 else max(_max(islots), _max(tslots))
    # end

    @property
    def input(self) -> list[int]:
        return self._islots

    @property
    def target(self) -> list[int]:
        return self._tslots
    
    def __getitem__(self, item):
        if item == 0:
            return self._islots
        else:
            return self._tslots

    def __len__(self) -> int:
        return self._len

    # end

    def __repr__(self):
        return f"slots[input={self._islots}, target={self._tslots}, len={len(self)}]"
# end


# ---------------------------------------------------------------------------
# LagResolver
# ---------------------------------------------------------------------------

class LagResolver:

    def __init__(self, lag, current=None):
        self._lag = lag
        self._normalize()
        self._validate()
        if current is not None:
            self._lag[LAG_CURRENT] = current
    # end

    @property
    def normalized(self) -> dict:
        return self._lag
    # end

    def _normalize(self):
        lag = self._lag
        if isinstance(lag, int):
            lag = dict(
                period_type=LAG_DAY,
                input=lag,
                target=lag
            )
        elif isinstance(lag, tuple):
            assert len(lag) == 2
            lag = dict(
                period_type=LAG_DAY,
                input=lag[0],
                target=lag[1]
            )

        assert isinstance(lag, dict), f"'{lag}' is not a valid lag configuration"

        # 'type' as alias of 'period_type'
        if 'type' in lag:
            lag[PERIOD_TYPE] = lag['type']
            del lag['type']

        if PERIOD_TYPE not in lag:
            lag[PERIOD_TYPE] = LAG_DAY

        if LAG_LENGTH in lag:
            period_type = lag[PERIOD_TYPE]
            lag_length = lag[LAG_LENGTH]
            del lag[LAG_LENGTH]
            lag[LAG_INPUT] = {period_type: lag_length}
            lag[LAG_TARGET] = {period_type: lag_length}

        assert LAG_INPUT in lag or LAG_TARGET in lag, \
            f"'{lag}' doesn't contain {LAG_INPUT} or {LAG_TARGET} entries"

        if LAG_INPUT in lag and LAG_TARGET not in lag:
            lag[LAG_TARGET] = lag[LAG_INPUT]
        if LAG_INPUT not in lag:
            lag[LAG_INPUT] = 0

        self._lag = lag

        period_type = lag[PERIOD_TYPE]
        lag[LAG_INPUT] = self._normalize_entry(period_type, lag[LAG_INPUT])
        lag[LAG_TARGET] = self._normalize_entry(period_type, lag[LAG_TARGET])

        self._lag = lag
    # end

    def _normalize_entry(self, period_type, lag):
        if isinstance(lag, int):
            return {
                period_type: lag
            }
        else:
            return lag
    # end

    def _validate(self):
        lag = self._lag
        assert PERIOD_TYPE in lag
        assert LAG_INPUT in lag
        assert LAG_TARGET in lag

        period_type = lag[PERIOD_TYPE]
        assert period_type in LAG_TYPES, f"'{period_type}' is not a valid period type"
        for lag_type in lag[LAG_INPUT]:
            assert lag_type in LAG_TYPES or isinstance(lag_type, int), f"'{lag_type}' is not a valid period type"
        for lag_type in lag[LAG_TARGET]:
            assert lag_type in LAG_TYPES or isinstance(lag_type, int), f"'{lag_type}' is not a valid period type"
    # end

    def resolve(self) -> LagSlots:
        current = True if LAG_CURRENT not in self._lag else self._lag[LAG_CURRENT]
        islots = self._resolve_entry(LAG_INPUT, current)
        tslots = self._resolve_entry(LAG_TARGET, current)
        return LagSlots(islots, tslots)
    # end

    def _resolve_entry(self, entry, current) -> set[int]:
        lag = self._lag[entry]
        period_type = self._lag[PERIOD_TYPE]
        base_factor = LAG_FACTORS[period_type]

        if entry == LAG_INPUT and current:
            slots = {0}
        else:
            slots = set()

        # lag is already a list of integers
        if isinstance(lag, (tuple, list)):
            slots.update(lag)
            return slots

        # trick to be sure that the ORDER of the lag types is from 'smaller' to 'larger'
        for lag_type in LAG_TYPES:
            if lag_type not in lag: continue

            lag_factor = LAG_FACTORS[lag_type]
            assert base_factor <= lag_factor, f"Lag {lag_type} in {entry} must be a multiple of period_type {period_type}"

            lag_size = lag_factor // base_factor
            lag_repl = lag[lag_type]

            if lag_repl >= 0:
                slots.update([i * lag_size for i in range(1, lag_repl + 1)])
            else:
                slots = set()
        # end
        # resolve the lag_type of integer type
        for lag_type in lag:
            if not isinstance(lag_type, int):
                continue
            lag_size = lag_type
            lag_repl = lag[lag_type]
            if lag_repl >= 0:
                slots.update([i * lag_size for i in range(1, lag_repl + 1)])
            else:
                slots = set()
        # end

        return slots
    # end
# end


# ---------------------------------------------------------------------------
# resolve_lag
# ---------------------------------------------------------------------------

def resolve_lag(lag: Union[int, tuple, dict], current=None) -> LagSlots:
    """
    Resolve the 'lag' configuration in a list of 'slots' to select in the input dataset, where a 'slot'
    is a record.

    The complete configuration is a dictionary having the following structure:

        {
            'period_type': <period_type>,
            'input': {
                <period_type_1>: <count_1>,
                <period_type_2>: <count_2>,
                ...
            },
            'target: {
                <period_type_1>: <count_1>,
                <period_type_2>: <count_2>,
                ...
            },
            'current': True
        }

    where

        - <period_type> (and <period_type_i>) is one of the following values:

            'second'    = 1
            'minute'    = 60 seconds
            'hour'      = 60 minutes
            'day'       = 24 hours
            'week'      = 7 days
            'month'     = 30 days
            'quarter'   = 91 days  (because 365//4 = 91 is a little better than 30*3 = 90)
            'year'      = 365 days

        - <count_i> is an integer value.

        - <current> is a flag used to obtain the same behaviour of sktime (skip the 'current' day)

    The main rule is that <period_type_i> MUST BE equals or grater than the reference <period_type>

    If not specified, the default value for 'period_type' is 'day'

    It is possible to specify only 'input', in this case the configuration is replicated to the other
    entry ('input' -> 'target'). Instead, if it is specified only 'target' entry, the 'input' entry
    will be empty

    If 'lag' is a single integer value, it is equivalent to:

        {
            'period_type': 'day',
            'input': {
                'day': <value>
            },
            'target': {
                'day': <value>
            }
        }

    If 'lag' is two integer values, it is equivalent to:

        {
            'period_type': 'day',
            'input': {
                'day': <values[0]>
            },
            'target': {
                'day': <values[1]>
            }
        }

    :param lag: the lag configuration
        1. a single integer value
        2. two integer values
        3. a dictionary
    :return LagSlots: an object containing the time slots to select for the input and target features.
    """
    lr = LagResolver(lag, current)
    res: LagSlots = lr.resolve()
    return res
# end


# ---------------------------------------------------------------------------
# LagTrainTransformer
# LagPredictTransform
# ---------------------------------------------------------------------------
# X can be None or X[n, 0]

class LagTrainTransform:

    def __init__(self, slots: LagSlots):
        assert isinstance(slots, LagSlots)
        self._slots: LagSlots = slots

    def fit(self, X: Optional[ndarray], y: np.ndarray):
        assert isinstance(y, np.ndarray)
        assert isinstance(X, (type(None), np.ndarray))
        if X is not None: assert len(y) == len(X)
        return self

    def transform(self, X: Optional[ndarray], y: np.ndarray):
        assert isinstance(y, np.ndarray)
        assert isinstance(X, (type(None), np.ndarray))

        slots: LagSlots = self._slots

        islots = list(reversed(slots.input))
        tslots = list(reversed(slots.target))
        s = len(slots)

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        # create a X[n,0] if X is None
        # and uses an empty list as islots
        if X is None:
            n = len(y)
            X = np.zeros((n, 0))
            islots = []

        assert len(y) == len(X)

        sx = len(islots)
        sy = len(tslots)
        n = y.shape[0]
        my = y.shape[1]
        mx = X.shape[1]

        nt = n - s
        mt = sx * mx + sy
        Xt = np.zeros((nt, mt))
        yt = np.zeros((nt, my))

        for i in range(nt):
            c = 0
            for j in tslots:
                Xt[i, c:c + my] = y[s + i - j]
                c += my
            for j in islots:
                Xt[i, c:c + mx] = X[s + i - j]
                c += mx
            yt[i] = y[s + i]
        # end

        return Xt, yt

    def fit_transform(self, X: Optional[ndarray], y: np.ndarray):
        return self.fit(X=X, y=y).transform(X=X, y=y)
# end


class LagPredictTransform:

    def __init__(self, slots: LagSlots):
        assert isinstance(slots, LagSlots)

        # Xh, yh: history (past)
        #   Xh can be None
        # Xp, yp: prediction (future)
        #   yp is created if not already passed
        #   Xp can be None
        # Xt: temporary input matrix used to generate yt (a SINGLE value)
        #   yt MUST be saved in yp ad the correct index

        self._slots: LagSlots = slots
        self._yh: Optional[ndarray] = None
        self._Xh: Optional[ndarray] = None
        # --
        self._Xp: Optional[ndarray] = None
        self._yp: Optional[ndarray] = None
        # --
        self._Xt: Optional[ndarray] = None
        # --
        self._m = 0
    # end

    def fit(self, X: Optional[ndarray], y: Optional[ndarray]):
        assert isinstance(y, np.ndarray)
        assert isinstance(X, (type(None), np.ndarray))

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        # X_history, y_history
        self._Xh = X
        self._yh = y

        slots = self._slots
        sx = len(slots.input)
        sy = len(slots.target)
        m = 0 if X is None else X.shape[1]

        nt = 1
        mt = sx*m + sy

        self._m = m
        self._Xt = np.zeros((nt, mt))
        return self

    def transform(self, X: Optional[ndarray] = None, y: Optional[ndarray] = None, fh: int = 0):
        assert isinstance(fh, int)
        assert y is not None or fh > 0

        if y is None:
            my = self._yh.shape[1]
            y = np.zeros((fh, my))

        # X_pred, y_pred
        self._Xp = X
        self._yp = y

        return y
    # end

    def fit_transform(self, X: Optional[ndarray] = None, y: Optional[ndarray] = None, fh: int = 0):
        return self.fit(X=X, y=y).transform(X=X, y=y, fh=fh)

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    def _atx(self, index):
        return self._Xh[index] if index < 0 else self._Xp[index]

    def _aty(self, index):
        return self._yh[index] if index < 0 else self._yp[index]

    def prepare(self, i: int):
        atx = self._atx
        aty = self._aty

        slots = self._slots
        m = self._m
        Xt = self._Xt

        if m == 0:
            c = 0
            for j in reversed(slots.target):
                Xt[0, c] = aty(i - j)
                c += 1
        else:
            c = 0
            for j in reversed(slots.target):
                Xt[0, c] = aty(i - j)
                c += 1
            for j in reversed(slots.input):
                Xt[0, c:c + m] = atx(i - j)
                c += m
        # end
        return Xt
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
