

class P1:
    def __init__(self):
        print("init P1")

    def say(self):
        print("I am P1")

    def say1(self):
        print("I am P1")


class P2:
    def __init__(self):
        print("init P2")

    def say(self):
        print("I am P2")

    def say2(self):
        print("I am P2")


class C(P1, P2):
    def __init__(self):
        # super().__init__()
        P1.__init__(self)
        P2.__init__(self)
        print("init C")

    def say(self):
        # super().say()
        # self.say1()
        # self.say2()
        # self.say3()
        P1.say(self)
        P2.say(self)
        print("I am C")

    def say3(self):
        print("I am P3")



c = C()
c.say()

