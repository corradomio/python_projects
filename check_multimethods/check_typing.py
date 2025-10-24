from typing import Collection

from multimethod import  multimethod


@multimethod
def fun(li: Collection[int]):
    return "list[int]"


@multimethod
def fun(li: Collection[str]):
    return "list[str]"


print(fun([1,2]))
print(fun(["str"]))
print(fun({1, 2}))
print(fun({"str"}))
print(fun((1, 2)))
print(fun(("str")))

