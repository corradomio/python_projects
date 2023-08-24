from typing import Union

__all__ = ['LagSlots', 'LagResolver', 'resolve_lag']

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAG_INPUT = 'input'
LAG_TARGET = 'target'
LAG_DAY = 'day'
TYPE = 'type'
PERIOD_TYPE = 'period_type'
LAG_LENGTH = 'length'
LAG_CURRENT = 'current'
DAY_SECONDS = 3600 * 24

LAG_FACTORS: dict[str, int] = {
    '': 0,

    'second': 1,
    'minute': 60,
    'hour': 3600,
    'day': DAY_SECONDS,
    'week': DAY_SECONDS * 7,
    'month': DAY_SECONDS * 30,
    'quarter': DAY_SECONDS * 91,
    'year': DAY_SECONDS * 365,

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
    def __init__(self, islots: set[int], tslots: set[int]):
        self._islots: list[int] = sorted(islots)
        self._tslots: list[int] = sorted(tslots)

        def _max(l: Union[list[int], set[int]]) -> int:
            return 0 if len(l) == 0 else max(l)

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

    # def __init__(self, lag, current=None):
    #     self._lag = lag
    #     self._normalize()
    #     self._validate()
    #     if current is not None:
    #         self._lag[LAG_CURRENT] = current
    # end
    def __init__(self, lag):
        assert isinstance(lag, dict)

        self._lag = {} | lag
        self._normalize()
        self._validate()

    # end

    @property
    def normalized(self) -> dict:
        return self._lag

    # end

    def _normalize(self):
        lag = self._lag

        # if isinstance(lag, int):
        #     lag = dict(
        #         period_type=LAG_DAY,
        #         input=lag,
        #         target=lag
        #     )
        # elif isinstance(lag, (tuple, list)):
        #     if len(lag) == 1:
        #         lag = (0, lag[0])
        #     assert len(lag) == 2
        #     lag = dict(
        #         period_type=LAG_DAY,
        #         input=lag[0],
        #         target=lag[1]
        #     )

        assert isinstance(lag, dict), f"'{lag}' is not a valid lag configuration"

        # 'type' as alias of 'period_type'
        if TYPE in lag:
            lag[PERIOD_TYPE] = lag[TYPE]
            del lag[TYPE]

        if PERIOD_TYPE not in lag:
            lag[PERIOD_TYPE] = LAG_DAY

        if LAG_LENGTH in lag:
            period_type = lag[PERIOD_TYPE]
            lag_length = lag[LAG_LENGTH]
            del lag[LAG_LENGTH]
            if isinstance(lag_length, int):
                lag[LAG_INPUT] = {period_type: lag_length}
                lag[LAG_TARGET] = {period_type: lag_length}
            elif len(lag_length) == 2:
                lag[LAG_INPUT] = {period_type: lag_length[0]}
                lag[LAG_TARGET] = {period_type: lag_length[1]}
            else:
                raise f"Invalid lag 'length': {lag_length}"
            # end
        # end

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
            if lag_type not in lag:
                continue

            lag_factor = LAG_FACTORS[lag_type]
            assert base_factor <= lag_factor, \
                f"Lag {lag_type} in {entry} must be a multiple of period_type {period_type}"

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

def resolve_lag(lag: Union[int, tuple, list, dict], current=None) -> LagSlots:
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

    Note 1: there is no check on the reference 'period type' and the period used in the datetime column.

    Note 2: parameter 'current', if not None, overrides the entry 'current' in the dictionary

    :param lag: the lag configuration
        1. a single integer value
        2. two integer values
        3. a dictionary
    :param current: if to consider the current time slot
    :return LagSlots: an object containing the time slots to select for the input and target features.
    """
    assert isinstance(lag, (int, tuple, list, dict)), "'lag' is not int, (int, int), or dict"

    if isinstance(lag, (int, tuple, list)):
        lag = {
            'length': lag,
            'current': current
        }
    elif current is not None:
        lag['current'] = current

    lr = LagResolver(lag)
    res: LagSlots = lr.resolve()
    return res
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
