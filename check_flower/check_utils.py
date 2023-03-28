from collectionx import bag
from annotations import method


class C:
    def __init__(self):
        self.c = 0

c = C()


@method(C)
def do(self):
    self.c += 1
    print(self.c)


c.do()

# print(bag([1,1,2,2,3,3,4,5,6]))
