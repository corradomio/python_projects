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

class LagSlots:
    def __init__(self, islots: set[int], tslots: set[int]):
        self._islots: list[int] = sorted(islots)
        self._tslots: list[int] = sorted(tslots)

        # self._len = 0 if len(tslots) == 0 else max(max(islots), max(tslots)) + 1
        self._len = 0 if len(tslots) == 0 else max(max(islots), max(tslots))

    # end

    @property
    def input_slots(self) -> list[int]:
        return self._islots

    @property
    def target_slots(self) -> list[int]:
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

    def __init__(self, lag):
        self._lag = lag
        self._normalize()
        self._validate()

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
        current = True if 'current' not in self._lag else self._lag['current']
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

def resolve_lag(lag: Union[int, tuple, dict]) -> LagSlots:
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
                period_type_1>: <count_1>,
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
    lr = LagResolver(lag)
    res: LagSlots = lr.resolve()
    # return res.input_slots, res.target_slots
    return res
# end


# ---------------------------------------------------------------------------
# LagTrainTransformer
# ---------------------------------------------------------------------------

class LagTrainTransform:

    def __init__(self, slots: LagSlots):
        assert isinstance(slots, LagSlots)
        self._slots: LagSlots = slots

    def fit(self, y: np.ndarray, X: Optional[ndarray]):
        assert isinstance(y, np.ndarray)
        assert isinstance(X, (type(None), np.ndarray))
        return self

    def transform(self, y: np.ndarray, X: Optional[ndarray]):
        assert isinstance(y, np.ndarray)
        assert isinstance(X, (type(None), np.ndarray))
        if X is not None: assert len(y) == len(X)

        slots: LagSlots = self._slots
        s = len(slots)

        sx = len(slots.input_slots)
        sy = len(slots.target_slots)
        n = y.shape[0]
        m = 0 if X is None else X.shape[1]

        nt = n - s
        mt = sx * m + sy
        Xt = np.zeros((nt, mt))
        yt = np.zeros(nt)

        if X is None:
            for i in range(nt):
                c = 0
                for j in reversed(slots.target_slots):
                    Xt[i, c] = y[s + i - j]
                    c += 1
                yt[i] = y[s + i]
            # end
        else:
            for i in range(nt):
                c = 0
                for j in reversed(slots.target_slots):
                    Xt[i, c] = y[s + i - j]
                    c += 1
                for j in reversed(slots.input_slots):
                    Xt[i, c:c + m] = X[s + i - j]
                    c += m
                yt[i] = y[s + i]
            # end

        return Xt, yt.reshape((-1, 1))

    def fit_transform(self, y: np.ndarray, X: Optional[ndarray]):
        return self.fit(y, X).transform(y, X)


class LagPredictTransform:

    def __init__(self, slots: LagSlots):
        assert isinstance(slots, LagSlots)

        self._slots: LagSlots = slots
        self._yh: Optional[ndarray] = None
        self._Xh: Optional[ndarray] = None
        self._Xt: Optional[ndarray] = None
        # --
        self._Xp: Optional[ndarray] = None
        self._yp: Optional[ndarray] = None
        self._m = 0
    # end

    def fit(self, X: Optional[ndarray], y: Optional[ndarray]):
        # X_history, y_history
        assert isinstance(y, np.ndarray)
        assert isinstance(X, (type(None), np.ndarray))

        self._Xh = X
        self._yh = y

        slots = self._slots
        sx = len(slots.input_slots)
        sy = len(slots.target_slots)
        m = 0 if X is None else X.shape[1]

        nt = 1
        mt = sx*m + sy

        self._m = m
        self._Xt = np.zeros((nt, mt))
        return self

    def transform(self, X: Optional[ndarray] = None, y: Optional[ndarray] = None, at: int = 0):
        # X_pred, y_pred
        self._Xp = X
        self._yp = y
        return self.prepare(at)
    # end

    def atx(self, index):
        return self._Xh[index] if index < 0 else self._Xp[index]

    def aty(self, index):
        return self._yh[index] if index < 0 else self._yp[index]

    def prepare(self, at: int):
        atx = self.atx
        aty = self.aty

        i = at
        slots = self._slots
        m = self._m
        Xt = self._Xt

        if m == 0:
            c = 0
            for j in reversed(slots.target_slots):
                Xt[0, c] = aty(i - j)
                c += 1
        else:
            c = 0
            for j in reversed(slots.target_slots):
                Xt[0, c] = aty(i - j)
                c += 1
            for j in reversed(slots.input_slots):
                Xt[0, c:c + m] = atx(i - j)
                c += m
        # end
        return Xt
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
