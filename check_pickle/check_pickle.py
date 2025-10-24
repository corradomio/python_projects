from stdlib import picklex


def main():
    data = {
        0: "zero",
        1: "one"
    }

    picklex.dump(data, "test.pickle")
    obj = picklex.load("test.pickle")

    obj1 = dict(**obj)

    pass


if __name__ == "__main__":
    main()
