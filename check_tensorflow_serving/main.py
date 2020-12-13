import tensorflow_serving as tfs

class B:

    def __init__(self):
        pass

    def pinco(self):
        print("pinco")


class C(B):

    def __init__(self):
        self.f1 = 1
        self.f2 = 2
        super().__init__()

    def pinco(self):
        print("pinco")


def main():
    b = B()
    b.pinco()
    b.pinco("x")
    pass


if __name__ == "__main__":
    main()
