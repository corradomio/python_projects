class MyClass:
    def __init__(self):
        pass

    def __repr__(self):
        return "Ciccio Pasticcio"
# end


def main():
    o = MyClass()
    print(type(o).__name__)

    assert 1==0, [1,2,"tre", MyClass()]


if __name__ == "__main__":
    main()
