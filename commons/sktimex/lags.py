#
# Deprecated: it is not necessary to support this 'complex' configurations
# It is enough to support list of integers ONLY
#

__all__ = [
    'LagSlots',
    'LagsResolver',
    'resolve_lags',
    'flatten_max',
    'lmax'
]

from typing import Union, Any

from .utils import RangeType

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
# Utilities
# ---------------------------------------------------------------------------

def lmax(l: list[int]) -> int:
    """
    Maximum value in a list, 0 for empty lists
    """
    return 0 if len(l) == 0 else max(l)


def flatten_max(ll: list[list[int]]) -> int:
    """
    Maximum scanning recursively a list of lists of int
    """
    m = 0
    for l in ll:
        if len(ll) == 0:
            continue
        m = max(m, lmax(l))
    return m


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def _flatten(ll: list[list]) -> list:
    """
    Convert a list of list in a flatten list
    """
    f = set()
    for l in ll:
        f.update(l)
    return sorted(f)


def _resolve_lags(entry: str, lags: dict) -> list[list[int]]:
    assert isinstance(lags, dict)

    def _check_current():
        return 0 in lags and 0 < lags[0]

    slots: list[list[int]] = []
    current = _check_current()
    if entry == LAGS_INPUT and current:
        slots.append([0])

    for lag_size in lags:
        # skip lag_size == 0, used as 'current')
        if lag_size == 0:
            continue

        lag_repl = lags[lag_size]
        if isinstance(lag_repl, list):
            slots.append(lag_repl)
        elif isinstance(lag_repl, tuple):
            slots.append(list(lag_repl))
        elif lag_repl > 0:
            slots.append([i * lag_size for i in range(1, lag_repl + 1)])
    # end
    return slots


def _is_integer(s: str) -> bool:
    """Check if s can be converted as an integer"""
    try:
        ival = int(s)
        return True
    except:
        return False


# ---------------------------------------------------------------------------
# LagSlots
# ---------------------------------------------------------------------------

class LagSlots:

    def __init__(self, lags):
        # if lags is None:
        #     # X[0] -> y[0]
        #     lags = {
        #         'input': {0: 1},
        #         'target': {}
        #     }
        assert isinstance(lags, dict)
        assert LAGS_INPUT in lags
        assert isinstance(lags[LAGS_INPUT], dict)
        assert LAGS_TARGET in lags
        assert isinstance(lags[LAGS_TARGET], dict)
        assert LAGS_CURRENT not in lags

        self._lags = lags
        self._xlags_lists = []
        self._ylags_lists = []
        self._xlags = []
        self._ylags = []

        self._xlags_lists = _resolve_lags(LAGS_INPUT, lags[LAGS_INPUT])
        self._ylags_lists = _resolve_lags(LAGS_TARGET, lags[LAGS_TARGET])
        self._xlags = _flatten(self._xlags_lists)
        self._ylags = _flatten(self._ylags_lists)

        # self._len = max(lmax(self._xlags), lmax(self._ylags))

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def lags(self) -> dict:
        """Lags as normalized dictionary"""
        return self._lags

    # -- input/xlags

    @property
    def xlags(self) -> list[int]:
        return self._xlags

    # @property
    # def input(self) -> list[int]:
    #     """Flatten list of input lags"""
    #     return self._xlags

    # @property
    # def xlags_lists(self):
    #     """List of input lags organized by multiplier"""
    #     return self._xlags_lists

    # @property
    # def input_lists(self):
    #     """List of input lags organized by multiplier"""
    #     return self._xlags_lists

    # -- target/ylags

    @property
    def ylags(self) -> list[int]:
        return self._ylags

    # @property
    # def target(self) -> list[int]:
    #     """Flatten list of target lags"""
    #     return self._ylags

    # @property
    # def ylags_lists(self):
    #     """List of target lags organized by multiplier"""
    #     return self._ylags_lists

    # @property
    # def target_lists(self):
    #     """List of target lags organized by multiplier"""
    #     return self._ylags_lists

    # -- other properties

    def __len__(self) -> int:
        """Window length containing all lags"""
        # return self._len
        return max(lmax(self._xlags), lmax(self._ylags))

    def __getitem__(self, item):
        # self[0] -> input
        # self[1] -> target
        if item == 0:
            return self._xlags
        else:
            return self._ylags

    def __repr__(self):
        return f"slots[input={self.xlags}, target={self.ylags}, len={len(self)}]"
# end


# ---------------------------------------------------------------------------
# LagsResolver
# ---------------------------------------------------------------------------
# Resolve the lags definition in the standard format composed by a dictionary
#
#       {
#           'input': {
#           },
#           'target': {
#           }
#       }
#
# The parameter 'current' is encodes as:
#
#       {
#           'input': {
#               0: 0|1,
#           }
#       }

class LagsResolver:
    def __init__(self, lags):
        assert isinstance(lags, (int, tuple, list, dict)), "'lags' is not int | (int,int) | (int,int,bool) | dict"

        if isinstance(lags, (int, tuple, list)):
            lags = {
                'length': lags
            }

        # create a copy because 'lags' is modified'
        self._lags: dict[str, Any] = {} | lags
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
                lags[LAGS_INPUT] = {1: 0}
                lags[LAGS_TARGET] = {1: lags_length}
            elif len(lags_length) == 2:
                lags[LAGS_INPUT] = {1: lags_length[0]}
                lags[LAGS_TARGET] = {1: lags_length[1]}
            elif len(lags_length) == 3:
                lags[LAGS_INPUT] = {1: lags_length[0]}
                lags[LAGS_TARGET] = {1: lags_length[1]}
                lags[LAGS_CURRENT] = lags_length[2]
            else:
                raise f"'{lags}': invalid configuration, 'lags' must be: int | (int,int) | (int,int,bool) | dict(...)"
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

        if LAGS_INPUT in lags and isinstance(lags[LAGS_INPUT], int):
            repl = lags[LAGS_INPUT]
            lags[LAGS_INPUT] = {1: repl}

        if LAGS_TARGET in lags and isinstance(lags[LAGS_TARGET], int):
            repl = lags[LAGS_TARGET]
            lags[LAGS_TARGET] = {1: repl}

        if LAGS_CURRENT in lags:
            current = lags[LAGS_CURRENT]
            if current is not None:
                lags[LAGS_INPUT][0] = 1 if current else 0
            del lags[LAGS_CURRENT]
    # end

    def _normalize_entry(self, entry: str):
        lags = self._lags
        if isinstance(lags[entry], int):
            lags[entry] = {1: entry}

        lags_entry: dict = lags[entry]
        keys = list(lags_entry.keys())
        base_factor = LAG_FACTORS[self._period_type]
        for lag_type in keys:
            if isinstance(lag_type, int):
                continue

            elif _is_integer(lag_type):
                ilag_type = int(lag_type)
                lag_value = lags_entry[lag_type]
                del lags_entry[lag_type]
                lags_entry[ilag_type] = lag_value

            else:
                lag_value = lags_entry[lag_type]
                del lags_entry[lag_type]

                lag_factor = LAG_FACTORS[lag_type]//base_factor
                lags_entry[lag_factor] = lag_value
        # end
        lags[entry] = lags_entry
    # end

    def _validate(self):
        lags = self._lags
        assert LAGS_INPUT in lags
        assert LAGS_TARGET in lags
        assert LAGS_CURRENT not in lags

        for lag_type in lags[LAGS_INPUT]:
            assert isinstance(lag_type, int), f"'{lag_type}' is not a valid period type"
        for lag_type in lags[LAGS_TARGET]:
            assert isinstance(lag_type, int), f"'{lag_type}' is not a valid period type"
    # end

    def resolve(self):
        return LagSlots(self._lags)
# end


# ---------------------------------------------------------------------------
# resolve_lags
# ---------------------------------------------------------------------------

def resolve_lags(lags: Union[int, tuple, list, dict]) -> LagSlots:
    """
    Resolve the 'lags' configuration in a list of 'slots' to select in the input dataset, where a 'slot'
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

    If 'lags=<value>' it is equivalent to:

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

    If 'lags=(<value0>, <value1>)' it is equivalent to:

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

    Note 3: parameter 'current' can be configured also as:

        {
            'input': {
                0: 0|1
            }
        }

    :param lags: the lag configuration
        1. a single integer value
        2. two integer values
        3. a dictionary
    :return LagSlots: an object containing the timeslots to select for the input and target features.
    """
    assert lags is not None

    if isinstance(lags, LagSlots):
        return lags

    lr = LagsResolver(lags)
    res: LagSlots = lr.resolve()
    return res


def resolve_ilags(ilags: Union[int, tuple, list]) -> list[int]:
    """
    Resolve i)input lags (xlags & ylags).
    Note: the list must be ordered in increase way BUT processed in the
          opposite way
    """
    if isinstance(ilags, int):
        # [1, ilags]
        return list(range(1, ilags+1))
    elif isinstance(ilags, RangeType):
        return sorted(list(ilags))
    else:
        assert isinstance(ilags, (list, tuple))
        return sorted(list(ilags))


def resolve_tlags(tlags: Union[int, tuple, list]) -> list[int]:
    """
    Resolve t)arget lags.
    """
    if isinstance(tlags, int):
        # [0, tlags-1]
        return list(range(tlags))
    elif isinstance(tlags, RangeType):
        return sorted(list(tlags))
    else:
        assert isinstance(tlags, (list, tuple))
        return sorted(list(tlags))


def tlags_start(tlags: list[int]):
    n = len(tlags)
    for i in range(n):
        if tlags[i] >= 0:
            return i
    raise ValueError(f"Parameter 'tlags' doesnt' contain positive timeslots: {tlags}")

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
