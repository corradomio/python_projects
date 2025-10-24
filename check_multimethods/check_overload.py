from typing import overload, Any

multimethod=overload

@overload
def alpha(x: Any): ...

@overload
def alpha(x: Any, y: Any): ...

# @overload
def alpha(x):
    print("alpha/1", x)


# @overload
def alpha(x, y):
    print("alpha/2", x, y)


# -------------------------------------------

@overload
def say(what: int): ...

@overload
def say(what: float): ...

@overload
def say(what: str): ...


def say(what: int):
    print("say int:", what)


def say(what: float):
    print("say flt:", what)


def say(what: str):
    print("say str:", what)


@overload
def ciccio(): ...

@overload
def ciccio(i: int): ...

@overload
def ciccio(i: int, j: int): ...


def ciccio():
    print("ciccio")


def ciccio(i: int):
    print(f"ciccio[{i}]")


def ciccio(i: int, j: int):
    print(f"ciccio[{i},{j}]")


def main():
    alpha(1)
    # alpha(1,2)

    say(1)
    say(2.)
    say("3")

    ciccio()
    ciccio(1)
    ciccio(1,2)

# end


if __name__ == "__main__":
    main()
