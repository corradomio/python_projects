from typing import Any

import numpy as np

from stdlib.is_instance import is_instance


def main():
    z = np.zeros((1,2), dtype=np.float16)

    # assert is_instance(z,np.ndarray)
    # assert is_instance(z,np.ndarray[Any])
    # assert is_instance(z,np.ndarray[tuple[int,int]])
    # assert is_instance(z,np.ndarray[Any, np.dtype[np.float16]])
    # assert is_instance(z,np.ndarray[tuple[int,int], np.dtype[np.float16]])
    # assert is_instance(z, np.ndarray[(1,2), np.number])
    # assert is_instance(z, np.ndarray[(1,2), np.inexact])
    # assert is_instance(z, np.ndarray[(1,2), np.float16])
    # assert is_instance(z, np.ndarray[(1,2), np.dtype[np.float16]])
    assert not is_instance(z, np.ndarray[(1,2), np.float16])
    pass


if __name__ == "__main__":
    main()
