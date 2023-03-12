#
# Integer Quadratic Programming
#
from path import Path as path
import numpy as np
import numpyx as npx
import cvxpy as cp

np.set_printoptions(suppress=True, precision=4, linewidth=1024)
DATA_DIR="D:\\Projects_PhD\\sfun_evaluate\\iidxs_mat"
FILE_MAT="abalone_dt_bv.csv"


def load_mat(fname, n=0):
    m = np.loadtxt(fname, delimiter=",", dtype=float)
    # assert npx.is_symmetric(m) and npx.is_pos_def(m)
    if n > 0:
        m = m[0:n, 0:n]
    return m
# end


def mat_to_pii(m):
    n = len(m)
    ii = m.copy()
    for i in range(n): ii[i, i] = 0
    pi = np.zeros(n)
    for i in range(n): pi[i] = m[i, i]
    # interaction indices, power indices
    return pi, ii
# end


def quadratic_problem(m, v) -> tuple:
    """
        max(x) -1/2 x^T Q x - c^T x                 1/2 x^T P x + q^T x

        s.t.    Ax == a                             Ax == b
                Bx <= b                             Gx <= h
                x in {0,1}^n

    :param m: power/interaction indices matrix
    :param n: matrix size
    :param v: n. of views
    :return: Q, c, A, b
    """
    int_ = float

    def mk_Q(m, v):
        n = len(m)
        Ixi = m.copy()
        for i in range(n): Ixi[i, i] = 0
        nv = n * v
        Q = np.zeros((nv, nv), dtype=float)

        for i in range(0, v):
            vi = n * i
            Q[vi:vi + n, vi:vi + n] = Ixi
            for j in range(i + 1, v):
                vj = n * j
                Q[vi:vi + n, vj:vj + n] = -Ixi
                Q[vj:vj + n, vi:vi + n] = -Ixi
            # end
        # end
        return Q

    def mk_c(m, v):
        n = len(m)
        cxi = np.zeros(n, dtype=float)
        for i in range(n): cxi[i] = m[i, i]
        nv = n*v
        c = np.zeros(nv, dtype=float)

        for i in range(v):
            vi = n*i
            c[vi:vi + n] = cxi
        return c

    def mk_Aa(m, v):
        # Ax = 1
        n = len(m)
        I = np.identity(n, dtype=int_)
        A = np.zeros((n, n * v), dtype=int_)
        for i in range(v):
            vi = n * i
            A[0:n, vi:vi + n] = I
        a = np.ones(n, dtype=int_)
        return A, a

    def mk_Bb(m, v):
        # implementa -x <= 0 and x <= 1
        n = len(m)
        nv = n*v
        B = np.zeros((2*nv, nv), dtype=int_)
        b = np.zeros(2*nv, dtype=int_)
        for i in range(nv):
            B[00 + i, i] = -1
            B[nv + i, i] = +1
            b[00 + i] = 0
            b[nv + i] = 1
        return B, b

    Q = mk_Q(m, v)
    c = mk_c(m, v)
    A, a = mk_Aa(m, v)
    B, b = mk_Bb(m, v)
    return Q, c, A, a, B, b
# end


def solve_quadratic_problem(Q_, c_, A_, a_, B_, b_):
    """

    :param np.ndarray Q_:
    :param np.ndarray c_:
    :param np.ndarray A_:
    :param np.ndarray a_:
    :param np.ndarray B_:
    :param np.ndarray b_:
    :return:
    """
    n = len(Q_)

    # P_ = matrix(-Q)
    # q_ = matrix(-c.reshape((len(c), 1)))
    # A_ = matrix(A)
    # b_ = matrix(a.reshape((len(a), 1)))
    # G_ = matrix(B)
    # h_ = matrix(b.reshape((len(b), 1)))
    # sol = qp(P_, q_, G_, h_, A_, b_, solver=None, initvals=None)
    P = -Q_
    q = -c_
    G = B_
    h = b_
    A = A_
    b = a_

    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
                      [G @ x <= h,
                       A @ x == b])

    for gp in [None, False, True]:
        for qcp in [None, False, True]:
            print("---------------------")
            print("  ", gp, qcp)
            try:
                prob.solve(gp=gp, qcp=qcp)
            except Exception as e:
                print("  no", e)
    print("---------------------")
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)


    # pprint(sol)
    # print(sol['x'])
    # print(sol['y'])
    # print(sol['z'])
    pass


def main():
    fname = path(DATA_DIR).joinpath(FILE_MAT)
    m = load_mat(fname, 0)
    Q, c, A, a, B, b = quadratic_problem(m, 4)
    solve_quadratic_problem(Q, c, A, a, B, b)
# end


if __name__ == "__main__":
    main()
