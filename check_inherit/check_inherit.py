

class A:
    def __init__(self):
        super().__init__()
        print("A")

    def say(self):
        print("A.say")


class B(A):
    def __init__(self):
        super().__init__()
        print("B")

    def say(self):
        super().say()
        print("B.say")

    def say2(self):
        print("C.say2")


class C(A):
    def __init__(self):
        super().__init__()
        print("C")

    def say(self):
        super().say()
        print("C.say")

class D(C,B):
    def __init__(self):
        super().__init__()
        print("D")

    def say(self):
        super().say()
        print("D.say")

    def say2(self):
        super().say2()
        print("D.say2")

    def say3(self):
        B.say2(self)
        print("D.say3")

d=D()
print("---")
d.say()
print("---")
d.say2()
print("---")
d.say3()
print("---")
