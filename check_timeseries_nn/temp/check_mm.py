from multimethod import multimethod


@multimethod
def mul(i: int) -> int:
    return i


@multimethod
def mul(s: str) -> str:
    return s*3


print(mul(10))
print(mul("s"))
