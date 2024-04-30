#
# Extensions to 'json' standard package to use 'load' and 'dump' directly
# with a file path
#
import json
from json import dumps, loads

OPEN_ARGS = ['mode', 'buffering', 'encoding', 'errors', 'newline', 'closefd', 'opener']

#
# compatibility constants to permit a 'perfect' correspondence between JSON syntax and
# Python syntax
#
true = True
false = False
null = None


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
        return json.load(fp)


def dump(obj, file: str, **kwargs):
    if 'indent' not in kwargs:
        kwargs['indent'] = 4
    oargs, dargs = _open_dump_args(kwargs)
    with open(file, mode="w", **oargs) as fp:
        return json.dump(obj, fp, **dargs)


save = dump
