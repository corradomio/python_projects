#
# 2008 - Ding - Formulas for approximating pseudo-Boolean random variables
# 2010 - Ding - Transforms of pseudo-Boolean random variables
#
from numpy import ndarray, zeros, ones, diag
from numpy.linalg import solve, inv
from numpy.random import rand
from iset import *
from mathx import comb, iseq, sqrt


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def banzhaf_mu(n: int):
    """constant weights (Banzhaf like)"""
    p = 1 << n
    mu = ones(p)*1./p
    return mu
# end


def shapley_mu(n: int):
    """cardinal weigth (Shapley like)"""
    C = ones(n)*1/n
    return cardinal_mu(C)
# end


def cardinal_mu(C: ndarray):
    """cardinal weights"""
    assert iseq(sum(C), 1.)
    n = len(C)
    p = 1 << n
    mu = zeros(p)
    for S in range(p):
        s = icard(S)
        mu[S] = C[s]*1/comb(n, s)
    return mu
# end


def player_mu(P: ndarray) -> ndarray:
    """player-based weights"""
    n = len(P)
    p = 1 << n
    N = p - 1

    def prob(S):
        f = 1   # factor
        m = 1   # mask
        for i in range(n):
            if S & m:
                f *= P[i]
            else:
                f *= 1-P[i]
            m <<= 1
        return f
    # end

    mu = zeros(p)
    for S in ipowerset(N):
        mu[S] = prob(S)
    return mu
# end


# ---------------------------------------------------------------------------
# Random weights
# ---------------------------------------------------------------------------

def random_mu(n):
    p = 1 << n
    w = rand(p)
    return w/w.sum()
# end


def random_cardinal_mu(n):
    C = rand(n+1)
    return cardinal_mu(C/C.sum())
# end


def random_player_mu(n):
    P = rand(n)
    return P #player_mu(P)
# end


# ---------------------------------------------------------------------------
# 2008 - Ding - Formulas for approximating pseudo-Boolean random variables
# ---------------------------------------------------------------------------
# xi:   binary order
# mu:   binary order
# b:    lexicographic order
#

def _mubar(S: int, T: int, mu: ndarray) -> float:   # lexicographic order
    p = len(mu)
    N = p - 1
    U = iunion(S, T)
    mubar = 0.    # mu bar
    for R in isubsets(U, N):
        mubar += mu[R]
    return mubar
# end


def _mumatrix(mu: ndarray, order: int=-1) -> ndarray:   # lexicographic order
    p = len(mu)
    n = ilog2(p)
    l = ilexcount((n if order < 0 else order), n)
    M = zeros((l, l))
    for i in range(l):
        S = ilexset(i, n)
        for j in range(l):
            T = ilexset(j, n)
            M[i, j] = _mubar(S, T, mu)
    return M


def _mubeta(xi: ndarray, mu: ndarray, order: int=-1) -> ndarray:   # lexicographic order
    assert len(xi) == len(mu)
    p = len(xi)
    n = ilog2(p)
    l = ilexcount((n if order < 0 else order), n)
    N = p - 1
    beta = zeros(l)
    for i in range(l):
        S = ilexset(i, n)
        b = 0
        for T in isubsets(S, N):
            b += xi[T]*mu[T]
        beta[i] = b
    return beta
# end


def mu_solve(xi: ndarray, mu: ndarray, order: int=-1) -> ndarray:    # lexicographic order
    assert len(xi) == len(mu)
    M = _mumatrix(mu, order=order)
    beta = _mubeta(xi, mu, order=order)
    b = solve(M, beta)
    return b
# end


def mu_eval(S, xi, b):
    """
    :param S: set to evaluate
    :param xi: function (binary order)
    :param b: approximation factors (lexicographic order)
    :return:
    """
    p = len(xi)
    n = ilog2(p)
    o = ilexorder(len(b), n)
    f = 0
    for T in isubsets_lex(S, n=n, k=[o]):
        i = ilexidx(T, n)
        f *= b[i]*xi[T]
    return f
# end


mu_matrix = _mumatrix
mu_beta = _mubeta


# ---------------------------------------------------------------------------
# 2010 - Ding - Transforms of pseudo-Boolean random variables
# ---------------------------------------------------------------------------

def _pmatrixz(mu: ndarray, order:int=-1) -> ndarray: # lexicographic order
    def x(S, i):
        return 1 if iismember(S, i) else 0

    def zixj(zi, xj, n):
        zs = ilexset(zi, n)
        xs = ilexset(xj, n)

        z = 1
        for i in imembers(zs):
            pi = mu[i]
            si = 1/sqrt(pi*(1-pi))
            z *= (x(xs, i) - pi)*si
        return z

    n = len(mu)
    l = ilexcount((n if order < 0 else order), n)
    m = zeros((l, l))
    for zi in range(l):
        for xj in range(l):
            m[zi,xj] = zixj(zi, xj, n)
    return m
# end


def _pdiagonal(mu, order:int=-1):   # lexicographic order
    def w(S, i):
        return mu[i] if iismember(S, i) else 1 - mu[i]

    def _eval(zi, n):
        zs = ilexset(zi, n)
        d = 1
        for i in range(n):
            d *= w(zs, i)
        return d

    n = len(mu)
    l = ilexcount((n if order < 0 else order), n)
    d = zeros(l)
    for zi in range(l):
        d[zi] = _eval(zi, n)
    return diag(d)
# end


def _pfunction(xi: ndarray, order: int=-1):  # lexicographic order
    p = len(xi)
    n = ilog2(p)
    l = ilexcount((n if order < 0 else order), n)
    fz = zeros(l)

    for i in range(l):
        S = ilexset(i, n)
        fz[i] = xi[S]
    return fz
# end


def _papprox(mu: ndarray, order=-1):

    n = len(mu)
    l = ilexcount((n if order < 0 else order), n)
    m = zeros((l, l))

    # zi = (xi - pi)/sqrt(pi*(1-pi))) = xi*si - pi*si, si = 1/sqrt(pi*(1-pi))

    #     |  0       i
    #   --+-------------------
    #   0 | 1 ... -pi*si ...
    #     | :        :
    # i+1 | 0       si   ...
    #     | :        :
    #

    m[0, 0] = 1
    for j in range(n):
        pi = mu[j]
        si = 1/sqrt(pi*(1-pi))
        m[0, j+1] = -pi*si
        m[j+1, j+1] = si
    # end

    for j in range(n+1, l):
        T = ilexset(j, n)
        T1, T2 = isplit(T)
        j1 = ilexidx(T1, n)
        j2 = ilexidx(T2, n)
        for S1 in isubsets(T1):
            for S2 in isubsets(T2):
                S = iunion(S1, S2)

                i1 = ilexidx(S1, n)
                i2 = ilexidx(S2, n)
                i = ilexidx(S, n)

                m[i, j] += m[i1,j1]*m[i2, j2]
            pass
        # end
    # end
    return m
# end


def _vapprox(v: ndarray, n, order=-1):
    l = ilexcount((n if order < 0 else order), n)
    if l == len(v):
        return v
    else:
        return v[0:l]
# end


def p_solve(xi: ndarray, mu: ndarray, order: int=-1) -> ndarray:  # lexicographic order
    n = len(mu)
    mz = _pmatrixz(mu)
    w = _pdiagonal(mu)
    v = _pfunction(xi)
    zv = mz.dot(w).dot(v)

    mx = _papprox(mu, order=order)
    za = _vapprox(zv, n, order=order)
    xa = mx.dot(za)
    return xa
# end


p_matrixz = _pmatrixz
p_approx = _papprox
p_diagonal = _pdiagonal
p_function = _pfunction


def p_check(P):
    w = _pdiagonal(P)
    m = _pmatrixz(P)
    t = m.T
    r = m.dot(w).dot(t).round(5)
    pass


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
