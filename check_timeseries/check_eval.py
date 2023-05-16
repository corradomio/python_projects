

class C:
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b
        pass

    def eval(self, exp, **kwargs):
        a = self.a
        b = self.b
        try:
            return eval(exp)
        except:
            return exp



def main():

    c = C(a=11, b=12)
    print(c.eval('a'))
    print(c.eval('b'))
    print(c.eval('x'))
    print(c.eval('y', x=13, y=14))
    print(c.eval('ciccio'))
    pass



if __name__ == "__main__":
    main()
