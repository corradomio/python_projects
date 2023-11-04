class Base:
    def __init__(self):
        # print("Base")
        pass

    # def say(self):
    #     print("say Base")


class Mixin:

    def say(self, super_):
        super_.say()
        print("say Mixin")


class Left(Base):
    def __init__(self):
        super().__init__()
        # print("Left")

    # def say(self):
    #     super().say()
    #     print("say Left")


class Right(Base):
    def __init__(self):
        super().__init__()
        # print("Right")

    # def say(self):
    #     super().say()
    #     print("say Right")


class Derived(Left, Right, Mixin):
    def __init__(self):
        super().__init__()
        print("Derived")

    # def say(self):
    #     super().say()
    #     print("say Derived")

    def say(self):
        super(Mixin).say(Mixin)



d = Derived()
d.say()

