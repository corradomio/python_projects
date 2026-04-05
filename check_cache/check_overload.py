import numpy as np
from multimethod import multimethod


@multimethod
def fun(x:np.ndarray[int]):
    return "np.array"

@multimethod
def fun(x: int) -> str:
    return "int"

@multimethod
def fun(x: float) -> str:
    return "float"

@multimethod
def fun(x: float, y:float) -> str:
    return "float/2"


@multimethod
def fun(x:list[int]):
    return "list[int]"


@multimethod
def fun(x:list[float]):
    return "list[float]"

print(fun(np.arange(10)))
print(fun(1))
print(fun(1.0))
print(fun(1.0, 1.0))
print(fun([1]))
print(fun([.1]))
