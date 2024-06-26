#
# Extensions to 'json' standard package:
#
#   1) to use 'load' and 'dump' directly with a file path
#   2) returning an 'stdlib.dict', a dictionary with a lot
#      of improvements useful for configurations
#
import json
from .dict import dict

OPEN_ARGS = ['mode', 'buffering', 'encoding', 'errors', 'newline', 'closefd', 'opener']


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def _open_dump_args(kwargs):
    oargs = {}
    dargs = {}
    for arg in kwargs:
        if arg in OPEN_ARGS:
            oargs[arg] = kwargs[arg]
        else:
            dargs[arg] = kwargs[arg]
    return oargs, dargs


def load(file: str, **kwargs) -> dict:
    with open(file, mode="r", **kwargs) as fp:
        return dict(json.load(fp))


def loads(s, **kwargs) -> dict:
    return dict(json.loads(s, **kwargs))


def dump(obj, file: str, **kwargs) -> dict:
    if 'indent' not in kwargs:
        kwargs['indent'] = 4
    oargs, dargs = _open_dump_args(kwargs)
    with open(file, mode="w", **oargs) as fp:
        return dict(json.dump(obj, fp, **dargs))


save = dump

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
