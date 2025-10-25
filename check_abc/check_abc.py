class A:
    def echo(self):
        print("A")

class B:
    def echo(self):
        print("B")

class C(A,B):
    def echo(self):
        super(B).echo()
        print("C")



c = C()
c.echo()

