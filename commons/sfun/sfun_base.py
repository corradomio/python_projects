#
#
#
from numpy import ndarray, array, loadtxt, savetxt, zeros, ones
from iset import *
from stdlib.mathx import sq, sqrt, pow, INF
from stdlib.imathx import qfact, comb, bernoulli, icomb, m1pow


# ---------------------------------------------------------------------------
# Numpy functions
# ---------------------------------------------------------------------------

ShapeType = Union[int, list[int], tuple[int]]


def fzeros(n: ShapeType) -> ndarray: return zeros(n, dtype=float)


def fones(n: ShapeType) -> ndarray: return ones(n, dtype=float)


# ---------------------------------------------------------------------------
# Utilities functions
# ---------------------------------------------------------------------------

def to_dict(xi: ndarray):
    d = dict()
    p = len(xi)
    N = p - 1

    for S in ipowerset(N):
        d[ilist(S)] = xi[S]
    return d
# end


def load_data(fname):
    """Create the function from the file"""
    data = loadtxt(fname, delimiter=",")
    if len(data.shape) == 2:
        nr, nc = data.shape
        vdata = fzeros(nr+1)
        for i in range(nr):
            vdata[int(data[i, 0])] = data[i, 1]
        data = vdata
    return data


def save_data(fname, data):
    savetxt(fname, data, delimiter=",")


# ---------------------------------------------------------------------------
# Other functions
# ---------------------------------------------------------------------------

def normalize(xi: ndarray) -> ndarray:
    """
    Normalize the function:

        min(xi) = 0 (generally xi[0])
        max(xi) = 1 (generally xi[N])

    :param xi:
    :return:
    """
    l = min(xi)
    u = max(xi)
    return (xi-l)/(u-l)
# end


# def conjugate_value(xi: ndarray, S: int) -> float:
#     """
#     Compute the conjugate value:
#
#         cf(A) = xi(N) - xi(N-A)
#
#     :param xi: set function values
#     :param S: set
#     :return:
#     """
#     n = len(xi)             # number of subsets
#     N = n - 1               # full set
#
#     C = idiff(N, S)
#     return xi[N] - xi[C]
# # end


def conjugate(xi: ndarray) -> ndarray:
    """
    Compute the conjugatea function

        cf(A) = xi(N) - xi(N-A)

    :param xi: set function
    :return: conjugate function
    """
    p = len(xi)             # number of subsets
    N = p - 1               # full set

    def _value(S: int) -> float:
        D = idiff(N, S)
        return xi[N] - xi[D]

    c = fzeros(p)  # conjugate function
    for S in ipowerset(N):
        c[S] = _value(S)
    return c
# end


# def derivative_value(xi: ndarray, S: int, K: int) -> float:
#     d = 0.
#     D = idiff(S, K)
#     for L in isubsets(K):
#         T = iunion(D, L)
#         d += sign(K, L) * xi[T]
#     return d
# # end


def derivative(xi: ndarray, K: int) -> ndarray:
    """
    Compute the derivative of the set function respect K
    If A is None compute the derivative of the set function, otherwise
    the derivative on A

    :param xi: set function
    :param K: indices for derivation
    :param A: reference set
    :return: the derivate of the function
    """
    p = len(xi)             # number of subsets
    k = icard(K)
    N = p - 1               # full set

    def _value(S: int) -> float:
        v = 0.
        D = idiff(S, K)
        for L in isubsets(K):
            T = iunion(D, L)
            v += idsign(K, L) * xi[T]
        return v

    d = fzeros(p)
    for S in ipowerset(N):
        d[S] = _value(S)
        # d[S] = derivative_value(xi, S, K)
    return d
# end


def reduce_with_respect_on(xi: ndarray, P: int) -> ndarray:
    """
    Reduced function

    :param xi: set function
    :param P: amalgamated set
    :return: the reduces function
    """
    p = len(xi)
    n = ilog2(p)
    N = p - 1

    L = ilist(P)
    l = len(L)

    r = fzeros(p-l+1)
    for S in ipowerset(N):
        T = ireduceset(S, P, n, L)
        if ihasintersect(S, P):
            r[T] = xi[iunion(S, P)]
        else:
            r[T] = xi[S]
    return r
# end


def induced_by(xi: ndarray, S: int) -> ndarray:
    s = icard(S)
    p = 1 << s
    ixi = fzeros(p)

    i = 0
    for T in isubsets(S):
        ixi[i] = xi[T]
    return ixi
# end


# ---------------------------------------------------------------------------
# Dot products
# ---------------------------------------------------------------------------

def weighted_dot(f: ndarray, g: ndarray, mu: ndarray) -> float:
    N = len(f) - 1
    return sum(f[S]*g[S]*mu[S] for S in ipowerset(N))
# end


def weighted_norm(f: ndarray, g: ndarray, mu: ndarray) -> float:
    N = len(f) - 1
    return sqrt(sum(sq(f[S] - g[S])*mu[S] for S in ipowerset(N)))
# end


# ---------------------------------------------------------------------------
# Additive Function
# ---------------------------------------------------------------------------

# def additive_value(sv, S) -> float:
#     return 0. + sum(sv[i] for i in imembers(S))
# # end


def compose_additive(sv) -> ndarray:
    n = len(sv)
    p = 1 << n
    N = p - 1
    xi = fzeros(p)

    def _value(S):
        return 0. + sum(sv[i] for i in imembers(S))

    for S in ipowerset(N):
        xi[S] = _value(S)
    return xi
# end


# ---------------------------------------------------------------------------
# Monotone Cover
# ---------------------------------------------------------------------------

# def monotone_cover_value(xi: ndarray, S: int) -> float:
#     m = 0.
#     for T in isubsets(S):
#         if xi[T] > m:  m = xi[T]
#     return m
# # end


# def monotone_cover(xi: ndarray) -> ndarray:
#     """
#     Compute the monotone cover
#
#     :param xi: set function
#     :param A: reference set
#     :return: the monotone cover
#     """
#     p = len(xi)             # number of subsets
#     N = p - 1               # full set
#
#     def _value(S: int) -> float:
#         m = 0.
#         for T in isubsets(S):
#             if xi[T] > m:  m = xi[T]
#         return m
#
#     mc = fzeros(p)
#     for S in ipowerset(N):
#         mc[S] = _value(S)
#         # mc[S] = monotone_cover_value(xi, S)
#     return mc
# # end


def monotone_cover(xi: ndarray) -> ndarray:
    """
    Compute the monotone cover

    :param xi: set function
    :param A: reference set
    :return: the monotone cover
    """
    p = len(xi)  # number of subsets
    N = p - 1  # full set

    def _value(S):
        v = mc[S]
        for i in imembers(S):
            Si = isub(S, i)
            if mc[Si] > v:
                v = mc[Si]
        return v

    mc = xi.copy()
    for S in ipowerset(N):
        # for i in imembers(S):
        #     Si = isub(S, i)
        #     if mc[Si] > mc[S]:
        #         mc[S] = mc[Si]
        mc[S] = _value(S)
    return mc
# end


# ---------------------------------------------------------------------------
# Mobius Transform
# ---------------------------------------------------------------------------
#
#   mobius_xi(A) = SUM(  B <= A : (-1)^|A-B| * xi(B) )
#
#   mobius_xi(A+i) = SUM( B <= A : (-1)^|A-B| * xi(B) ) +
#                    SUM( B <= A : (-1)^|A-B| * xi(B + i))
#                  = mobius_xi(A) + SUM( B <= A : (-1)^|A-B|+1 * xi(B + i))
#

# def mobius_value(xi: ndarray, S: int) -> float:
#     return sum(sign(S, T) * xi[T] for T in isubsets(S))
# # end


def mobius_transform(xi: ndarray, n_jobs=None) -> ndarray:
    """
    Compute the mobius transform of the set function

    :param xi: set function
    :return: the mobius transform
    """
    p = len(xi)             # number of subsets
    N = p - 1               # full set
    # m = fzeros(p)

    def _value(S):
        v = sum(idsign(S, T) * xi[T] for T in isubsets(S))
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    m = array(results)
    return m
# end


def inverse_mobius_value(m: ndarray, S: int) -> float:
    return sum(m[T] for T in isubsets(S))
# end


def inverse_mobius_transform(m: ndarray, n_jobs=None) -> ndarray:
    """
    Recompose the set function from its mobius transform

    :param m: set function in mobius format
    :return: the recomposed function
    """
    p = len(m)              # number of subsets
    N = p - 1               # full set

    def _value(S):
        return sum(m[T] for T in isubsets(S))

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end

# ---------------------------------------------------------------------------
# Walsh Transform
# ---------------------------------------------------------------------------


def walsh_value(xi: ndarray, S: int):
    p = len(xi)
    N = p-1

    v = sum(idsign(S, T) * xi[T] for T in ipowerset(N))/p
    return v
# end


def walsh_transform(xi: ndarray, n_jobs=None) -> ndarray:
    p = len(xi) # 2^n
    N = p - 1

    def _value(S):
        v = sum(idsign(S, T) * xi[T] for T in ipowerset(N))/p
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    wt = array(results)
    return wt
# end


def inverse_walsh_value(wt: ndarray, S: int) -> float:
    N = len(wt) - 1
    return sum(idsign(T, S) * wt[T] for T in ipowerset(N))
# end


def inverse_walsh_transform(wt: ndarray, n_jobs=None) -> ndarray:
    p = len(wt)
    N = p - 1

    def _value(S):
        v = sum(idsign(T, S) * wt[T] for T in ipowerset(N))
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


# ---------------------------------------------------------------------------
# Shapley Value
# ---------------------------------------------------------------------------

def shapley_value(xi: ndarray, i: int) -> float:
    p = len(xi)             # size of the powerset
    n = ilog2(p)            # n of elements
    N = p - 1               # full set
    Ni = isub(N, i)

    sv = 0.
    for S in isubsets(Ni):
        s = icard(S)
        Si = iadd(S, i)
        sv += qfact(n, s) * (xi[Si] - xi[S])
    return sv
# end


def shapley_values(xi: ndarray):
    p = len(xi)             # size fo the powerset
    n = ilog2(p)            # n of elements
    N = p - 1               # full set
    sv = fzeros(n)          # Shapley Value

    def _value(i):
        Ni = isub(N, i)

        v = 0.
        for S in isubsets(Ni):
            s = icard(S)
            Si = iadd(S, i)
            v += qfact(n, s) * (xi[Si] - xi[S])
        return v

    for i in range(n):
        sv[i] = _value(i)
        # sv[i] = shapley_value(xi, i)
    return sv
# end


# def shapley_value_correction(xi, sv):
#     return pvalue_correction(xi, sv)
# #end


def shapley_transform(xi: ndarray, n_jobs=None) -> ndarray:
    p = len(xi)
    n = ilog2(p)
    N = p - 1
    # st = fzeros(p)

    def _value(S):
        s = icard(S)
        NS = idiff(N, S)
        v = 0.
        for T in isubsets(NS):
            t = icard(T)
            v += qfact(n, t, s) * sum(idsign(S, K) * xi[iunion(K, T)] for K in isubsets(S))
        return v

    # resuls = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    st = array(results)
    return st
# end


def inverse_shapley_transform_value(st: ndarray, S: int) -> float:
    p = len(st)
    N = p - 1

    def b(K):
        l = icard(K)
        k = icard(iinterset(S, K))
        return 0. + sum(comb(k, j) * bernoulli(l - j) for j in range(k + 1))

    return 0. + sum(b(K) * st[K] for K in ipowerset(N))
# end


def inverse_shapley_transform(st: array, n_jobs=None) -> ndarray:
    p = len(st)
    N = p - 1

    def b(S, K):
        l = icard(K)
        k = icard(iinterset(S, K))
        return sum(comb(k, j) * bernoulli(l - j) for j in range(k + 1))

    def _value(S):
        return 0. + sum(b(S, K) * st[K] for K in ipowerset(N))

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N)
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


# ---------------------------------------------------------------------------
# Banzhaf Value
# ---------------------------------------------------------------------------

def banzhaf_value(xi: ndarray, i: int) -> float:
    p = len(xi)             # size of the powerset
    N = p - 1               # full set
    Ni = isub(N, i)

    bv = 0.
    c = 0
    for S in isubsets(Ni):
        Si = iadd(S, i)
        bv += (xi[Si] - xi[S])
        c += 1
    return bv/c
# end


def banzhaf_values(xi: ndarray) -> (ndarray, ndarray):
    p = len(xi)             # size fo the powerset
    n = ilog2(p)            # n of elements
    N = p - 1               # full set
    bv = fzeros(n)          # Banzhaf Value

    def _value(i):
        Ni = isub(N, i)

        v = 0.
        c = 0.
        for S in isubsets(Ni):
            Si = iadd(S, i)

            v += (xi[Si] - xi[S])
            c += 1
        # return 1. / 2 ** (n - 1) * bv
        return v/c

    for i in range(n):
        bv[i] = _value(i)
        # bv[i] = banzhaf_value(xi, i)
    return bv
# end


def banzhaf_transform(xi: ndarray, n_jobs=None) -> ndarray:
    p = len(xi)
    n = ilog2(p)
    N = p - 1
    # bi = fzeros(p)

    def _value(S):
        s = icard(S)
        v = sum(idsign(S, K) * xi[K] for K in ipowerset(N))
        return pow(.5, (n - s)) * v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    bi = array(results)
    return bi
# end


def inverse_banzhaf_transform_value(bt: ndarray, S: int) -> float:
    p = len(bt)
    N = p - 1

    v = 0.
    for T in ipowerset(N):
        t = icard(T)
        v += pow(.5, t) * idsign(T, S) * bt[T]
    return v
# end


def inverse_banzhaf_transform(bt: ndarray, n_jobs=None) -> ndarray:
    p = len(bt)
    N = p - 1

    def _value(S):
        v = 0.
        for T in ipowerset(N):
            t = icard(T)
            v += pow(.5, t) * idsign(T, S) * bt[T]
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


# def shapley_interaction_index(xi: ndarray) -> ndarray:
#     p = len(xi)             # size fo the powerset
#     n = ilog2(p)            # n of elements
#     N = p - 1       # full set
#     iv = fzeros((n, n))     # Shapley Interaction Value
#
#     def _value(i, j):
#         Nij = idiff(N, iset([i, j]))
#         v = 0.
#         for T in isubsets(Nij):
#             t = icard(T)
#             Ti = iadd(T, i)
#             Tj = iadd(T, j)
#             Tij = iunion(Ti, Tj)
#
#             t = qfact(n, t, 2) * (xi[Tij] - xi[Ti] - xi[Tj] + xi[T])
#             v += t
#         return v
#
#     for i in range(n - 1):
#         for j in range(i + 1, n):
#             iv[i, j] = _value(i, j)
#             # iv[i, j] = shapley_interaction_value(xi, i, j)
#
#     for i in range(n-1):
#         for j in range(i+1, n):
#             iv[j, i] = iv[i, j]
#
#     return iv
# # end


# ---------------------------------------------------------------------------
# Chaining from Shapley Transform
# ---------------------------------------------------------------------------

# def shapley_value(xi: ndarray, S: int) -> float:
#     p = len(xi)
#     n = ilog2(p)
#     s = icard(S)
#     N = p - 1
#     NS = idiff(N, S)
#
#     v = 0.
#     for T in isubsets(NS):
#         t = icard(T)
#         # q = 0.
#         # for K in isubsets(S):
#         #     q += sign(S, K)*xi[iunion(K, T)]
#         # v += qfact(p, t, s)*q
#         v += qfact(n, t, s)*sum(sign(S, K)*xi[iunion(K, T)] for K in isubsets(S))
#     return v
# # end


def chaining_from_shapley_transform(st):
    p = len(st)
    N = p - 1
    ct = fzeros(p)

    def _coeff(s, t):
        return sum(comb(t-s, k-s)*s/k*bernoulli(t-k) for k in range(s, t+1))

    def _value(S):
        if S == 0: return 0.
        s = icard(S)
        v = 0.
        for T in isubsets(S, N):
            t = icard(T)
            v += _coeff(s, t)*st[T]
        return v

    for S in ipowerset(N):
        ct[S] = _value(S)
    return ct
# end


# ---------------------------------------------------------------------------
# Mobius Transform from Shapley Transform
# ---------------------------------------------------------------------------

# def mobius_from_shapley_value(m: ndarray, S: int) -> float:
#     p = len(m)
#     N = p - 1
#     s = icard(S)
#
#     mv = 0.
#     for T in isubsets(S, N):
#         t = icard(T)
#         mv += 1/(t-s+1)*m[T]
#     return mv
# # end


def shapley_from_mobius_transform(m: ndarray) -> ndarray:
    p = len(m)
    N = p - 1
    st = fzeros(p)

    def _value(S):
        s = icard(S)
        v = 0.
        for T in isubsets(S, N):
            t = icard(T)
            v += 1 / (t - s + 1) * m[T]
        return v

    for S in ipowerset(N):
        st[S] = _value(S)
        # st[S] = mobius_from_shapley_value(m, S)
    return st
# end


def shapley_pi_from_transform(st: ndarray) -> ndarray:
    p = len(st)
    n = ilog2(p)
    pi = fzeros(n)
    for i in range(n):
        pi[i] = st[iset(i)]
    return pi
# end


def shapley_ii_from_transform(st: ndarray) -> ndarray:
    p = len(st)
    n = ilog2(p)
    ii = fzeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ii[i, j] = ii[j, i] = st[iset([i, j])]
    return ii
# end


# ---------------------------------------------------------------------------
# Banzhaf Transform
# ---------------------------------------------------------------------------

# def banzhaf_transform_value(xi: ndarray, S: int) -> float:
#     p = len(xi)
#     n = ilog2(p)
#     N = p - 1
#     s = icard(S)
#
#     v = sum(sign(S, K)*xi[K] for K in ipowerset(N))
#     return pow(.5, (n-s))*v
# # end


def banzhaf_pi_from_transform(bt: ndarray) -> ndarray:
    p = len(bt)
    n = ilog2(p)
    pi = fzeros(n)
    for i in range(n):
        pi[i] = bt[iset(i)]
    return pi
# end


def banzhaf_ii_from_transform(bt: ndarray) -> ndarray:
    p = len(bt)
    n = ilog2(p)
    ii = fzeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ii[i, j] = ii[j, i] = bt[iset([i, j])]
    return ii
# end


# ---------------------------------------------------------------------------
# Mobius from Banzhaf
# ---------------------------------------------------------------------------

# def mobius_from_banzhaf_value(m: ndarray, S: int) -> float:
#     p = len(m)
#     N = p-1
#     s = icard(S)
#
#     mv = 0.
#     for T in isubsets(S, N):
#         t = icard(T)
#         mv += pow(0.5, t-s)*m[T]
#     return mv
# # end


def banzhaf_from_mobius_transform(m: ndarray) -> ndarray:
    p = len(m)
    N = p-1
    bt = fzeros(p)

    def _value(S):
        s = icard(S)
        v = 0.
        for T in isubsets(S, N):
            t = icard(T)
            v += pow(0.5, t - s) * m[T]
        return v

    for S in ipowerset(N):
        bt[S] = _value(S)
        # bt[S] = mobius_banzhaf_value(m, S)
    return bt
# end


# def banzhaf_fixed_approx(bt: ndarray, k: int) -> ndarray:
#     p = len(bt)
#     N = p - 1
#     ba = fzeros(p)
#
#     for S in ipowerset(N):
#         if icard(S) <= k:
#             ba[S] = bt[S]
#     return ba
# # end


# ---------------------------------------------------------------------------
# co-Mobius
# ---------------------------------------------------------------------------

# def comobius_value(xi: ndarray, A: int) -> float:
#     p = len(xi)
#     N = p - 1
#     return sum(sign(B) * xi[idiff(N, B)] for B in isubsets(A))
# # end


def comobius_transform(xi: ndarray, n_jobs=None) -> ndarray:
    p = len(xi)
    N = p - 1
    # cm = fzeros(p)

    def _value(S):
        v = sum(idsign(B) * xi[idiff(N, B)] for B in isubsets(S))
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    cm = array(results)
    return cm
# end


def inverse_comobius_value(cm: ndarray, A: int) -> float:
    p = len(cm)
    N = p - 1
    D = idiff(N, A)
    return sum(idsign(B) * cm[B] for B in isubsets(D))
# end


def inverse_comobius_transform(cm: ndarray, n_jobs=None) -> ndarray:
    p = len(cm)
    N = p - 1

    def _value(S):
        D = idiff(N, S)
        return sum(idsign(B) * cm[B] for B in isubsets(D))

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


# ---------------------------------------------------------------------------
# Fourier Transform
# ---------------------------------------------------------------------------

# def fourier_value(xi: ndarray, S: int) -> float:
#     p = len(xi)
#     n = ilog2(p)
#     N = p - 1
#     return 1./p*sum(msign(S, T)*xi[T] for T in ipowerset(N))
# # end


def fourier_transform(xi: ndarray, n_jobs=None) -> ndarray:
    p = len(xi)
    N = p - 1
    # ft = fzeros(p)

    def _value(S):
        v = 1. / p * sum(iisign(S, T) * xi[T] for T in ipowerset(N))
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    ft = array(results)
    return ft
# end


def inverse_fourier_value(ft: ndarray, S: int) -> float:
    p = len(ft)
    N = p - 1
    return sum(iisign(S, T) * ft[T] for T in ipowerset(N))
# end


def inverse_fourier_transform(ft: ndarray, n_jobs=None) -> ndarray:
    p = len(ft)
    N = p - 1

    def _value(S):
        return sum(iisign(S, T) * ft[T] for T in ipowerset(N))

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


# ---------------------------------------------------------------------------
# Chaining Transform
# ---------------------------------------------------------------------------

def chaining_value(xi: ndarray, S: int) -> float:
    if S == 0: return 0.
    p = len(xi)
    n = ilog2(p)
    N = p - 1
    s = icard(S)
    cv = 0.
    for T in isubsets(N):
        U = iunion(S, T)
        u = icard(U)
        cv += (s * idsign(S, T)) / (n * comb(n - 1, u - 1)) * xi[T]
    return cv
# end


def chaining_transform(xi: ndarray, n_jobs=None) -> ndarray:
    p = len(xi)
    n = ilog2(p)
    N = p - 1

    def _value(S):
        if S == 0: return 0.
        s = icard(S)
        v = 0.
        for T in isubsets(N):
            U = iunion(S, T)
            u = icard(U)
            v += (s * idsign(S, T)) / (n * comb(n - 1, u - 1)) * xi[T]
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    ct = array(results)
    return ct
# end


def inverse_chaining_value(ct: ndarray, S: int) -> float:
    p = len(ct)
    N = p - 1
    fv = 0.
    D = idiff(N, S)
    for T in isubsets(D):
        t = icard(T)
        ts = 0.
        for i in imembers(S):
            Ti = iadd(T, i)
            ts += ct[Ti]
        # end
        fv += (m1pow(t)) / (t + 1) * ts
    return fv
# end


def inverse_chaining_transform(ct: ndarray, n_jobs=None) -> ndarray:
    p = len(ct)
    N = p - 1

    def b(S, T):
        if T == 0: return 0.
        t = icard(T)
        TS = iinterset(T, S)
        ts = icard(TS)
        return 1/t*sum(comb(ts, j)*m1pow(t-j)*j for j in range(ts+1))

    def _value(S):
        # v = 0.
        # for T in isubsets(N):
        #     v += b(S, T)*ct[T]
        v = sum(b(S, T)*ct[T] for T in isubsets(N))
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


def chaining_from_mobius_transform(m: ndarray) -> ndarray:
    p = len(m)
    N = p - 1
    ct = fzeros(p)

    def _value(S):
        s = icard(S)
        v = 0.
        for T in isubsets(S, N):
            t = icard(T)
            v += s/t*m[T]
        return v

    for S in ipowerset(N, empty=False):
        ct[S] = _value(S)
    return ct
# end


def mobius_from_chaining_trasform(ct):
    p = len(ct)
    N = p - 1
    m = fzeros(p)

    def _value(S):
        s = icard(S)
        v = 0.
        for T in isubsets(S, N):
            t = icard(T)
            v += m1pow(t-s)*s/t*ct[T]
        return v

    for S in ipowerset(N, empty=False):
        m[S] = _value(S)
    return m
# end


def shapley_from_chaining_trasform(ct):
    p = len(ct)
    N = p - 1
    st = fzeros(p)

    def _value(S):
        s = icard(S)
        v = ct[S]
        for T in isubsets(S, N, lower=False):
            t = icard(T)
            v += m1pow(t-s)*(s-1)/(t*(t-s+1))*ct[T]
        return v

    for S in ipowerset(N):
        st[S] =_value(S)
    return st
# end


def banzhaf_from_chaining_trasform(ct):
    p = len(ct)
    N = p - 1
    bt = fzeros(p)

    def _value(S):
        s = icard(S)
        v = 0.
        for T in isubsets(S, N):
            t = icard(T)
            if t == 0:
                v += ct[T]
            else:
                v += ipow(-.5, t-s)*(2*s-t)/t*ct[T]
        return v

    for S in ipowerset(N):
        bt[S] =_value(S)
    return bt
# end


# ---------------------------------------------------------------------------
# Player Probabilistic Transform
# ---------------------------------------------------------------------------

def player_probabilistic_value(xi: ndarray, S: int, mu: ndarray) -> float:
    p = len(xi)
    N = p - 1

    def _prod(T):
        f = 1.
        for i in imembers(idiff(T, S)):
            f *= mu[i]
        for i in imembers(idiff(N, iunion(S, T))):
            f *= (1 - mu[i])
        return f

    pv = 0.
    for T in ipowerset(N):
        pv += idsign(S, T) * xi[T] * _prod(T)
    return pv
# end


def player_probabilistic_transform(xi: ndarray, mu: ndarray, n_jobs=None) -> ndarray:
    p = len(xi)
    N = p - 1

    def _value(S):
        return player_probabilistic_value(xi, S, mu)

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    pt = array(results)
    return pt
# end


def inverse_player_probabilistic_value(pt: ndarray, S: int, mu: ndarray) -> float:
    def _prod(T):
        f = 1.
        for i in imembers(T):
            f *= imember(S, i) - mu[i]
        return f

    p = len(pt)
    N = p - 1
    fv = 0.
    for T in ipowerset(N):
        fv += pt[T]*_prod(T)
    return fv
# end


def inverse_player_probabilistic_transform(pt: ndarray, mu: ndarray, n_jobs=None) -> ndarray:
    p = len(pt)
    N = p - 1

    def _prod(S, T):
        f = 1.
        for i in imembers(T):
            f *= imember(S, i) - mu[i]
        return f

    def _value(S):
        v = 0.
        for T in ipowerset(N):
            v += pt[T] * _prod(S, T)
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


def mobius_from_player_probabilistic_value(pt: ndarray, S: int, mu: ndarray) -> float:
    p = len(pt)
    N = p - 1

    def _weight(S, T):
        D = idiff(T, S)
        w = 1.
        for i in imembers(D):
            w *= -mu[i]
        return w

    mv = 0.
    for T in isubsets(S, N):
        mv += pt[T] * _weight(S, T)
    return mv
# end


def mobius_from_player_probabilistic_transform(pt: ndarray, mu: ndarray) -> ndarray:
    p = len(pt)
    N = p - 1
    m = fzeros(p)

    def _weight(S, T):
        D = idiff(T, S)
        w = 1.
        for i in imembers(D):
            w *= -mu[i]
        return w

    def _value(S):
        v = 0.
        for T in isubsets(S, N):
            v += pt[T]*_weight(S, T)
        return v

    for S in ipowerset(N):
        m[S] = _value(S)
    return m
# end


def player_probabilistic_from_mobius_value(m: ndarray, S: int, mu: ndarray) -> float:
    p = len (m)
    N = p - 1

    def _weight(S, T):
        D = idiff(T, S)
        w = 1.
        for i in imembers(D):
            w *= mu[i]
        return w

    pv = 0.
    for T in isubsets(S, N):
        pv += m[T] * _weight(S, T)
    return pv
# end


def player_probabilistic_from_mobius_transform(m: ndarray, mu: ndarray) -> ndarray:
    p = len(m)
    N = p - 1
    pt = fzeros(p)

    def _weight(S, T):
        D = idiff(T, S)
        w = 1.
        for i in imembers(D):
            w *= mu[i]
        return w

    def _value(S):
        v = 0.
        for T in isubsets(S, N):
            v += m[T]*_weight(S, T)
        return v

    for S in ipowerset(N):
        pt[S] = _value(S)
    return pt
# end


def ppt_change_weights(pt, mu, nmu):
    p = len(pt)
    N = p - 1
    npt = fzeros(p)

    def _weight(S, T):
        D = idiff(T, S)
        w = 1.
        for i in imembers(D):
            w *= (nmu[i] - mu[i])
        return w

    def _value(S):
        v = 0.
        for T in isubsets(S, N):
            v += _weight(S, T)*pt[T]
        return v

    for S in ipowerset(N):
        npt[S] = _value(S)
    return npt
# end


def cpt_change_weights(pt, mu, nmu):
    pass


# ---------------------------------------------------------------------------
# Cardinal Probabilistic Transform
# ---------------------------------------------------------------------------

def cardinal_probabilistic_value(xi, mu, S):
    pass


def cardinal_probabilistic_transform(xi, mu, n_jobs=None):
    p = len(xi)
    n = ilog2(p)
    N = p - 1

    def _value(S):
        s = icard(S)
        NS = idiff(N, S)
        v = 0.
        for T in isubsets(NS):
            if T == 0: continue
            t = icard(T)
            v += mu[t-1]*icomb(n-s, t) * sum(idsign(S, K) * xi[iunion(K, T)] for K in isubsets(S))
        return v

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    cpt = array(results)
    return cpt
# end


def inverse_cardinal_probabilistic_value(pt, S, mu):
    pass


def inverse_cardinal_probabilistic_transform(pt, mu, n_jobs=None):
    p = len(pt)
    N = p - 1

    def b(K):
        l = icard(K)
        k = icard(iinterset(S, K))
        return 0. + sum(comb(k, j) * bernoulli(l - j) for j in range(k + 1))

    def _value(K):
        return 0. + sum(b(K) * pt[K] for K in ipowerset(N))

    # results = Parallel(n_jobs=n_jobs)(delayed(_value)(S) for S in ipowerset(N))
    results = [_value(S) for S in ipowerset(N)]
    xi = array(results)
    return xi
# end


def mobius_from_cardinal_probabilistic_transform(pt, mu):
    pass


# ---------------------------------------------------------------------------
# Marginal Data
# ---------------------------------------------------------------------------

def marginal_data_at(xi: ndarray, kmin: int, kmax: int, i: int) -> list:
    p = len(xi)
    N = p - 1  # N = {0,1...}
    Ni = isub(N, i)  # Ni = N - {i}

    mi = []
    for k in range(kmin, kmax + 1):
        for S in ilexsubset(Ni, k=k):
            Si = iadd(S, i)

            # marginal data
            mi.append(xi[Si] - xi[S])

    return mi
# end


def marginal_data(xi: ndarray, kmin: int, kmax: int) -> ndarray:
    p = len(xi)
    n = ilog2(p)

    md = []
    for i in range(n):
        mdi = marginal_data_at(xi, kmin, kmax, i)
        md.append(mdi)

    return array(md).T
# end


# ---------------------------------------------------------------------------
# Best worst set
# ---------------------------------------------------------------------------
# Search the sets with size 0,1,2,... with the maximum valus
#

def best_set(xi, k) -> int:
    """
    Search the best set at level k

    :param xi:
    :param k:
    :return:
    """
    p = len(xi)
    N = p - 1

    M = 0
    m = -INF
    for S in ilexsubset(N, k=k):
        if xi[S] > m:
            m = xi[S]
            M = S
    return M
# end


def worst_set(xi, k) -> int:
    """
    Search the worst set ad lebel k

    :param xi:
    :param k:
    :return:
    """
    p = len(xi)
    N = p - 1

    M = 0
    m = +INF
    for S in ilexsubset(N, k=k):
        if xi[S] < m:
            m = xi[S]
            M = S
    return M
# end


# ---------------------------------------------------------------------------
# Find best/worst set
# ---------------------------------------------------------------------------

def is_best_set(xi: ndarray, S: int) -> bool:
    """
    Search the best set starting from S and changing only 1 element

    :param xi:
    :param S:
    :return:
    """
    p = len(xi)
    N = p - 1
    D = idiff(N, S)
    s = icard(S)

    m = xi[S]
    for i in imembers(S):
        for j in imembers(D):
            T = ireplace(S, i, j)
            t = icard(T)

            assert s == t

            if xi[T] > m:
                return False
    return True
# end


def is_worst_set(xi: ndarray, S: int) -> bool:
    """
    Search the worst set starting from S and changing only 1 element

    :param xi:
    :param S:
    :return:
    """
    p = len(xi)
    N = p - 1
    D = idiff(N, S)

    M = S
    m = xi[M]
    for i in imembers(S):
        for j in imembers(D):
            T = ireplace(S, i, j)
            if xi[T] < m:
                return False
    return True
# end


def best_set_transform(xi: ndarray) -> ndarray:
    """
    Create a new set function based on this rule:

    if S is the best set (changing ONLY a single element)
        return xi[S]
    else
        return 0

    :param xi:
    :return:
    """
    p = len(xi)
    n = ilog2(p)
    N = p - 1

    def _value(S):
        return xi[S] if is_best_set(xi, S) else 0.

    bst = fzeros(p)
    for S in ipowerset(N):
        bst[S] = _value(S)
    return bst
# end


def are_levels_valid(xi: ndarray) -> bool:
    """
    Check is each leve contains at minimum a value different by zero
    :param xi:
    :return:
    """

    p = len(xi)
    n = ilog2(p)
    N = p - 1
    nz = fzeros(n+1)

    for S in ipowerset(N, empty=False):
        if xi[S] > 0:
            s = icard(S)
            nz[s] = 1.

    return nz.sum() == n
# end


# ---------------------------------------------------------------------------
# K-Power Indices
# ---------------------------------------------------------------------------

def k_shapley_value(xi: ndarray, i: int, k: int) -> float:
    p = len(xi)             # size of the powerset
    n = ilog2(p)            # n of elements
    N = p - 1               # full set
    Ni = isub(N, i)

    sv = 0.
    for S in ilexsubset(Ni, k=k-1):
        s = icard(S)
        Si = iadd(S, i)
        sv += qfact(n, s) * (xi[Si] - xi[S])
    return sv
# end


def k_banzhaf_value(xi: ndarray, i: int, k: int) -> float:
    p = len(xi)             # size of the powerset
    N = p - 1               # full set
    Ni = isub(N, i)

    bv = 0.
    c = 0
    for S in ilexsubset(Ni, k=k-1):
        Si = iadd(S, i)
        bv += (xi[Si] - xi[S])
        c += 1
    return bv/c
# end


def k_chaining_value(xi: ndarray, i: int, k: int) -> float:
    p = len(xi)
    n = ilog2(p)
    N = p - 1
    s = 1
    S = iset([i])
    Ni = isub(N, i)
    cv = 0.
    for T in ilexsubset(Ni, k=k-1):
        U = iunion(T, S)
        u = icard(U)
        cv += (s * idsign(S, T)) / (n * comb(n - 1, u - 1)) * xi[T]
    return cv
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

