from typing import Union, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAGS_INPUT = 'input'
LAGS_TARGET = 'target'
LAGS_DAY = 'day'
TYPE = 'type'
PERIOD_TYPE = 'period_type'
LAGS_LENGTH = 'length'
LAGS_CURRENT = 'current'
DAY_SECONDS = 3600 * 24

LAG_FACTORS: dict[str, int] = {
    '': 0,

    'second':   1,
    'minute':   60,
    'hour':     3600,
    'day':      DAY_SECONDS,
    'week':     DAY_SECONDS * 7,
    'month':    DAY_SECONDS * 30,
    'quarter':  DAY_SECONDS * 91,
    'year':     DAY_SECONDS * 365,

    'S': 1,
    'M': 60,
    'H': 3600,
    'd': DAY_SECONDS,
    'w': DAY_SECONDS * 7,
    'm': DAY_SECONDS * 30,
    'q': DAY_SECONDS * 91,
    'y': DAY_SECONDS * 365,

    'D': DAY_SECONDS,
    'W': DAY_SECONDS * 7,
    'Q': DAY_SECONDS * 91,
    'Y': DAY_SECONDS * 365,
}

LAG_TYPES: list[str] = list(LAG_FACTORS)


# ---------------------------------------------------------------------------
# LagSlots
# ---------------------------------------------------------------------------

class LagSlots:
    def __init__(self, islots: list[int], tslots: list[int], lags=None):
        assert isinstance(islots, list)
        assert isinstance(tslots, list)

        self._lags = lags
        self._islots: list[int] = islots
        self._tslots: list[int] = tslots

        self._len = max(tslots) if len(islots) == 0 else max(max(islots), max(tslots))

    @property
    def lags(self):
        return self._lags

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

    def __repr__(self):
        return f"slots[input={self._islots}, target={self._tslots}, len={len(self)}]"
# end


# ---------------------------------------------------------------------------
# LagsResolver
# ---------------------------------------------------------------------------

class LagsResolver:
    def __init__(self, lags, current: Optional[bool] = None):
        assert isinstance(lags, (int, tuple, list, dict)), "'lag' is not int, (int, int), or dict"
        assert isinstance(current, (type(None), bool))

        if isinstance(lags, (int, tuple, list)):
            lags = {
                'length': lags,
                'current': current
            }
        elif current is not None:
            lags['current'] = current

        # create a copy because 'lags' is modified'
        self._lags = {} | lags
        self._period_type = None

        self._normalize()
        self._validate()
    # end

    @property
    def normalized(self) -> dict:
        return self._lags
    # end

    def _normalize(self):
        self._normalize_lags()
        self._normalize_entry(LAGS_INPUT)
        self._normalize_entry(LAGS_TARGET)
    # end

    def _normalize_lags(self):
        lags = self._lags

        if TYPE in lags:
            self._period_type = lags[TYPE]
            del lags[TYPE]
        if PERIOD_TYPE in lags:
            self._period_type = lags[PERIOD_TYPE]
            del lags[PERIOD_TYPE]
        if self._period_type is None:
            self._period_type = LAGS_DAY

        if LAGS_LENGTH in lags:
            lags_length = lags[LAGS_LENGTH]
            del lags[LAGS_LENGTH]

            if isinstance(lags_length, int):
                lags[LAGS_INPUT] = {1: lags_length}
                lags[LAGS_TARGET] = {1: lags_length}
            elif len(lags_length) == 2:
                lags[LAGS_INPUT] = {1: lags_length[0]}
                lags[LAGS_TARGET] = {1: lags_length[1]}
            else:
                raise f"'{lags}': invalid configuration, 'length' must be int or (int, int)"
        # end

        # check if it is present 'input' and or 'target'
        assert LAGS_INPUT in lags or LAGS_TARGET in lags, \
            f"'{lags}': doesn't contain {LAGS_INPUT} or {LAGS_TARGET} entries"

        # if only 'input', copy 'input' onto 'target'
        if LAGS_INPUT in lags and LAGS_TARGET not in lags:
            lags[LAGS_TARGET] = lags[LAGS_INPUT]

        # if only 'target', set 'input: 0'
        if LAGS_INPUT not in lags:
            lags[LAGS_INPUT] = {1: 0}
    # end

    def _normalize_entry(self, entry: str):
        lags = self._lags
        if isinstance(lags[entry], int):
            lags[entry] = { 1: entry }

        lags_entry: dict = lags[entry]
        keys = list(lags_entry.keys())
        base_factor = LAG_FACTORS[self._period_type]
        for lag_type in keys:
            if isinstance(lag_type, int):
                continue

            lag_value = lags_entry[lag_type]
            del lags_entry[lag_type]

            lag_factor = LAG_FACTORS[lag_type]//base_factor
            lags_entry[lag_factor] = lag_value

        lags[entry] = lags_entry
    # end

    def _validate(self):
        lags = self._lags
        assert LAGS_INPUT in lags
        assert LAGS_TARGET in lags

        for lag_type in lags[LAGS_INPUT]:
            assert isinstance(lag_type, int), f"'{lag_type}' is not a valid period type"
        for lag_type in lags[LAGS_TARGET]:
            assert isinstance(lag_type, int), f"'{lag_type}' is not a valid period type"
    # end

    def resolve(self) -> LagSlots:
        lags = self._lags
        islots = self._resolve_entry(LAGS_INPUT)
        tslots = self._resolve_entry(LAGS_TARGET)
        return LagSlots(islots, tslots, lags)
    # end

    def _resolve_entry(self, entry):
        current = True if LAGS_CURRENT not in self._lags else self._lags[LAGS_CURRENT]
        lags: dict = self._lags[entry]
        assert isinstance(lags, dict)

        if entry == LAGS_INPUT and current:
            slots = {0}
        else:
            slots = set()

        # lag is already a list of integers
        if isinstance(lags, (tuple, list)):
            slots.update(lags)
            return slots

        # resolve the lag_type of integer type
        for lag_type in lags:
            lag_size = lag_type
            lag_repl = lags[lag_type]
            if lag_repl >= 0:
                slots.update([i * lag_size for i in range(1, lag_repl + 1)])
            else:
                slots = set()
        # end

        return sorted(slots)
    # end
# end


# ---------------------------------------------------------------------------
# resolve_lag
# ---------------------------------------------------------------------------

def resolve_lags(lags: Union[int, tuple, list, dict], current=None) -> LagSlots:
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
            'current': <bool>
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
            'year'      = 365 days (because 365 is a little better than 12*30 = 360)

        - <count_i> is an integer value.

        - <current> is a boolean value (True|False) used to obtain the same behaviour of sktime (skip the 'current' day)
          that is, if to use the 'input' features of the current timeslot to predict the current 'target'

    Aliases for 'second' ... 'year' are Python 'datetime' codes

            'Y'|'y' year
            'm'     month
            'd'     day
            'W'|'w' week
            'Q'|'q' quarter
            'H'     hour
            'M'     minute
            'S'     second

    The main rule is that <period_type_i> MUST BE equals or grater than the reference <period_type>

    If not specified, the default value for 'period_type' is 'day'

    An alternative approach is to specify <period_type_i> as an integer, that is, the multiplicity
    of the reference 'period type'. In this way, it is independent on the reference 'period type'.

    It is possible to specify only 'input', in this case the configuration is replicated to the 'target'.
    Instead, if it is specified only 'target' entry, the 'input' entry will be empty ([] or [0], based on 'current')

    If 'lag=<value>' it is equivalent to:

        <value> ::=

        {
            'period_type': 'day',
            'input': {
                'day': <value>
            },
            'target': {
                'day': <value>
            },
            'current': True
        }

    or also

        {
            'input': {
                1: <value>
            },
            'target': {
                1: <value>
            },
            'current': True
        }

    If 'lag=(<value0>, <value1>)' it is equivalent to:

        (<value0>, <value1>) ::=

        {
            'period_type': 'day',
            'input': {
                'day': <value0>
            },
            'target': {
                'day': <value1>
            },
            'current': True
        }

    or also

        {
            'input': {
                1: <value0>
            },
            'target': {
                1: <value1>
            },
            'current': True
        }

    There are some other simplified formats:

        {
            'length': <value>
        }

    equivalent to

        {
            'input': {
                1: <value>
            },
            'target': {
                1: <value>
            },
            'current': True
        }

    or also

        {
            'input': <value>
        }

    equivalent to

        {
            'input': {
                1: <value>
            },
            'target': {
                1: <value>
            },
            'current': True
        }



    Note 1: there is no check on the reference 'period type' and the period used in the datetime column.

    Note 2: parameter 'current', if not None, overrides the entry 'current' in the dictionary

    :param lag: the lag configuration
        1. a single integer value
        2. two integer values
        3. a dictionary
    :param current: if to consider the current time slot
    :return LagSlots: an object containing the time slots to select for the input and target features.
    """
    lr = LagsResolver(lags, current)
    res: LagSlots = lr.resolve()
    return res
# end


resolve_lag = resolve_lags

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
