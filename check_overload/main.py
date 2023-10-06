from multimethod import overload


@overload
def say(what: int):
    print("int", what)


@overload
def say(what: str):
    print("str", what)


def main():
    say(101)
    say("La carica dei 101")


if __name__ == "__main__":
    main()



