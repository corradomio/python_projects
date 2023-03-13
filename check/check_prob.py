from probability import *
from itertoolsx import *


def pfmt(l, d=4):
    return [round(e, d) for e in l]

def test1():
    n = 10
    P = uniform(n)
    print("P", P, sum(P))
    print()
    print("pwset", prob_family(powersetn(n), P))
    print("     ", pfmt(prob_elements(powersetn(n), P)))
    print()
    print("lvl[0,n]", prob_family(subsetsn(n, 0, n), P))
    print("     ", pfmt(prob_elements(powersetn(n, 0, n), P)))
    print()
    for i in range(n + 1):
        print("lvl[{}]".format(i), prob_family(subsetsn(n, i), P))
        print("       ", pfmt(prob_elements(subsetsn(n, i), P)))
    print()
    for i in range(n + 1):
        print("lvl[0,{}]".format(i), prob_family(subsetsn(n, 0, i), P))
        print("       ", pfmt(prob_elements(subsetsn(n, 0, i), P)))


def test2():
    n = 10
    P = uniform(n)
    L = uniform(n+1)
    print("pwset", prob_family(powersetn(n), P))


def main():
    # test1()
    test2()
# end


if __name__ == "__main__":
    main()
