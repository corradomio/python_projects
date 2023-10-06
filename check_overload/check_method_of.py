from languagex import method_of


class C:
    def __init__(self):
        pass


@method_of(C)
def say(self, what: int):
    print(what)


c = C()
c.say("Ciao ciccio")
