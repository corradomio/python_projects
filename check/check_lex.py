from csvx import save_csv
from mathx import comb, isprime
from randomx import random_list
import numpy as np
from itertoolsx import lex_index, lex_set, bin_index, bin_set, bit_index, bit_set
from math import sqrt
from pprint import pprint
SQ2 = sqrt(2)


def matrix(P):
    def zlex(zi, xj, n):
        zs = lex_set(zi, n)
        xs = lex_set(xj, n)

        z = 1
        for i in zs:
            # z *= 2 * x(xs, i) - 1
            z *= (x(xs, i) - P[i])/sqrt(P[i]*(1-P[i]))
        return z
    n = len(P)
    p = 1 << n
    m = np.ones((p, p))
    for zi in range(p):
        for xj in range(p):
            m[zi,xj] = zlex(zi, xj, n)
    return m


def diagonal(P):
    def w(S, i):
        return P[i] if i in S else 1-P[i]

    def mu(zi, n):
        zs = lex_set(zi, n)
        d = 1
        for i in range(n):
            d *= w(zs, i)
        return d

    n = len(P)
    p = 1 << n
    m = np.zeros((p, p))
    for zi in range(p):
        m[zi,zi] = mu(zi, n)
    return m



def test1():
    n = 8
    print("Bit")
    for i in range(1 << n):
        s = bit_set(i, n)
        m = bit_index(s, n)
        print("  ", i, m, s)
    print("Lexicographic")
    for i in range(1 << n):
        s = lex_set(i, n)
        m = lex_index(s, n)
        print("  ", i, m, s)
    print("Binary")
    for i in range(1 << n):
        s = bin_set(i, n)
        m = bin_index(s, n)
        print("  ", i, m, s)
    print("End")



def x(S, i): return 1 if i in S else 0


def zlex(zi, xj, n):
    zs = lex_set(zi, n)
    xs = lex_set(xj, n)

    z = 1
    for i in zs:
        z *= 2*x(xs, i) - 1
    return z


def zbin(zi, xj, n):
    zs = bin_set(zi, n)
    xs = bin_set(xj, n)

    z = 1
    for i in zs:
        z *= (2*x(xs, i) - 1)
    return z


def z3(zi, xj, n):
    zs = lex_set(zi, n)
    xs = lex_set(xj, n)

    z = 1
    for i in zs:
        z *= (3*x(xs, i) - 2)/SQ2
    return z


def test2():
    n = 3
    p = 1 << n

    for zi in range(p):
        for xj in range(p):
            print("{0:3}".format(zlex(zi, xj, n)), " ", end="")
        print()
    print()
    for xj in range(p-1, -1, -1):
        for zi in range(p):
            print("{0:3}".format(zbin(zi, xj, n)), " ", end="")
        print()
    print()


def test3():
    n = 3
    p = 1 << n

    for zi in range(p):
        for xj in range(p):
            print("{0:7}".format(round(z3(zi, xj, n), 3)), " ", end="")
        print()
    print()


def test4():
    m = matrix([.5]*3)
    print(m)
    print()
    m = matrix([2/3]*3)
    with np.printoptions(precision=3, suppress=True):
        print(m)
    pass


def test5():
    n = 4
    # P = [.5] * n
    # P = [2 / 3] * n
    P = random_list(n, True)
    w = diagonal(P)
    m = matrix(P)
    t = m.T
    r = m.dot(w).dot(t).round(5)
    with np.printoptions(precision=3, suppress=True):
        print(r)
    with np.printoptions(precision=3, suppress=True, edgeitems=100, linewidth=10000):
        print(w)
    pass


def gen_table():
    n = 10
    p = 1 << n
    data = []
    for i in range(p):
        b = len(bin_set(i, n))
        l = len(lex_set(i, n))
        data.append([i, b, l])
    save_csv("cardinalities.csv", data, header=["i", "binary", "lexicographic"])


def main():
    # test1()
    # test2()
    # test3()
    # test4()
    test5()
    # gen_table()
    pass


if __name__ == "__main__":
    main()