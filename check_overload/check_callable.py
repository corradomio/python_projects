

def a_function(p1, p2):
    print("a_function", p1, p2)


class ACallable():
    def __init__(self):
        pass

    def __call__(self, p1, p2):
        print("a_callable", p1, p2)


a_function(1,2)

c = ACallable()
c(11, 22)

