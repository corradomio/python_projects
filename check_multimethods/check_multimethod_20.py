from typing import Any

from multimethod import multimethod


@multimethod
def alpha(x: Any):
    print("alpha 1", x)


@multimethod
def alpha(x: Any, y: Any):
    print("alpha 2", x, y)


@multimethod
def say(what: int):
    print("say int:", what)


@multimethod
def say(what: float):
    print("say flt:", what)


@multimethod
def say(what: str):
    print("say str:", what)


@multimethod
def ciccio():
    print("ciccio")


@multimethod
def ciccio(i: int):
    print(f"ciccio[{i}]")


@multimethod
def ciccio(i: int, j: int):
    print(f"ciccio[{i},{j}]")


def main():
    # say(1)
    # say(2.)
    # say("3")
    #
    # ciccio()
    # ciccio(1)
    # ciccio(1,2)

    alpha(1)
    alpha(1,2)
# end


if __name__ == "__main__":
    main()
