#
# 16/10/2019 Corrado Mio (Local)
#
from cvxopt import matrix, solvers
from pprint import pprint


def main():
    A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
    b = matrix([1.0, -2.0, 0.0, 4.0])
    c = matrix([2.0, 1.0])
    sol = solvers.lp(c, A, b)
    pprint(sol)


if __name__ == "__main__":
    main()

