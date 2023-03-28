import typing as typ
from typing import *

import numpy as np

from typing_grammar import parse, HintType


def type_args(t: type) -> list[type]:
    if hasattr(t, '__args__'):
        return t.__args__
    else:
        return []


T = typ.Union[int, float]
# T = Callable[[int, float, str, list[int]], str]
# T = TypeVar('T')
# T = list[int]

T = int | float
print(T)

res: HintType = parse(T)
res.print()


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

res: HintType = parse(LogRegParams)
res.print()

