from multimethod import multimethod, overload


@multimethod
def say(what: int):
    print("int:", what)


@multimethod
def say(what: float):
    print("flt:", what)


@multimethod
def say(what: str):
    print("str:", what)


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
    say(1)
    say(2.)
    say("3")
    ciccio()
    ciccio(1)
    ciccio(1,2)
# end


if __name__ == "__main__":
    main()
