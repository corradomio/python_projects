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


def main():
    say(1)
    say(2.)
    say("3")


if __name__ == "__main__":
    main()
