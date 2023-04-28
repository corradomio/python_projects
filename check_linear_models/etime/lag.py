from typing import Union

__all__ = ['LagSlots', 'LagResolver', 'resolve_lag']

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
# lag:
#   int,
#   (int, int),
#   dict:
#       {
#           'period_type': <period_type>
#           'input': {
#               <period_type>: <count>,
#               ...
#           },
#           'target': {
#               ...
#           }
#       }
#
# period_type: second, minute, hour, day, week, month, quarter, year
#

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAG_INPUT = 'input'
LAG_TARGET = 'target'
LAG_DAY = 'day'
PERIOD_TYPE = 'period_type'

LAG_FACTORS: dict[str, int] = {
    '': 0,
    'second': 1,
    'minute': 60,
    'hour': 3600,
    'day': 3600*24,
    'week': 3600*24*7,
    # '4weeks': 1*60*60*24*7*4,
    'month': 3600*24*30,
    # '3months': 3600*24*30*3,
    # '4months': 3600*24*30*4,
    'quarter': 3600*24*91,       # 91 = 365//4 is a little better than 90 = 30*3
    'year': 3600*24*365,
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

        assert LAG_INPUT in lag or LAG_TARGET in lag, f"'{lag}' doesn't contain {LAG_INPUT} or {LAG_TARGET} entries"

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
            assert lag_type in LAG_TYPES, f"'{lag_type}' is not a valid period type"
        for lag_type in lag[LAG_TARGET]:
            assert lag_type in LAG_TYPES, f"'{lag_type}' is not a valid period type"
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

        # trick to ke sore that the ORDER of the lag types is from 'smaller' to 'larger'
        for lag_type in LAG_TYPES:
            if lag_type not in lag: continue

            lag_factor = LAG_FACTORS[lag_type]
            assert base_factor <= lag_factor, f"Lag {lag_type} in {entry} must be a multiple of period_type {period_type}"

            lag_size = lag_factor//base_factor
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
# end
# ---------------------------------------------------------------------------
