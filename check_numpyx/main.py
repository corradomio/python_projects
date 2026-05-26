from typing import Any
from pprint import pprint

import numpy as np


# def ndarray_class_getitem_(cls, item: Any, /):
#     pass
#
# np.ndarray.__dict__["__class_getitem__"] = ndarray_class_getitem_


def main():
    # pprint(list.__class_getitem__)
    # pprint(list.__dict__)
    pprint(np.__version__)

    # pprint(np.ndarray.__dict__)
    pprint(np.ndarray[Any, np.dtype[np.float64]])
    pprint(np.ndarray[tuple[int], np.dtype[np.int64]])

# end


if __name__ == "__main__":
    main()
    pass
