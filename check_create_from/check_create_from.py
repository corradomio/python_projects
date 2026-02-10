import numpy as np
from typing import Optional

from stdlib import jsonx

# def to_py_types(v: dict) -> dict:
#
#     def repl(v):
#         if isinstance(v, dict):
#             return drepl(v)
#         if isinstance(v, list):
#             return lrepl(v)
#         if isinstance(v, np.integer):
#             return int(v)
#         if isinstance(v, np.inexact):
#             return float(v)
#         return v
#
#     def drepl(d: dict) -> dict:
#         for k in d:
#             d[k] = repl(d[k])
#         return d
#
#     def lrepl(l: list) -> list:
#         for i in range(len(l)):
#             l[i] = repl(l[i])
#         return l
#
#     return repl(v)

def to_py_types(v: dict) -> dict:

    def repl(v):
        if isinstance(v, dict):
            return drepl(v)
        if isinstance(v, list):
            return lrepl(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.inexact):
            return float(v)
        return v

    def drepl(d: dict) -> dict:
        return {
            k: repl(d[k])
            for k in d
        }

    def lrepl(l: list) -> list:
        return [
            repl(v) for v in l
        ]

    return repl(v)




def dict_repl(d: dict, r: Optional[dict]=None) -> dict:
    assert isinstance(d, dict)
    assert isinstance(r, dict|type(None))

    if r is None or len(r) == 0:
        return d

    def is_param(v):
        return isinstance(v, str) and v.startswith("{") and v.endswith("}")

    def has_param(v):
        return isinstance(v, str) and "{" in v and "}" in v

    def value_of(v: str):
        p = v[1:-1]
        return r[p]

    def insert_into(v: str):
        while("{" in v):
            bgn = v.find("{")
            end = v.find("}", bgn + 1)
            p = v[bgn+1:end]
            v = v[:bgn] + r[p] + v[end+1:]
        return v

    def repl(v):
        if isinstance(v, dict):
            return drepl(v)
        if isinstance(v, list):
            return lrepl(v)
        if isinstance(v, str):
            if is_param(v):
                return value_of(v)
            elif has_param(v):
                return insert_into(v)
        return v

    def drepl(d: dict) -> dict:
        return {
            k: repl(d[k])
            for k in d
        }

    def lrepl(l: list) -> list:
        return [
            repl(v)
            for v in l
        ]

    return repl(d)


def main():
    # config = jsonx.load("auto_models.json")
    #
    # config = dict_repl(config, {
    #     "p11":11,
    #     "p123": 123
    # })
    #
    # print(jsonx.dumps(config))

    data = to_py_types({
        "p1": np.int32(1),
        "p2": [
            np.int64(21),
            22
        ]
    })
    pass




if __name__ == "__main__":
    main()
