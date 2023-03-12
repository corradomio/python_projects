#
# Probability
#
# from random import random
# from itertoolsx import subsetsc
#
#
# def _prob(l):
#     p = 1
#     for f in l: p *= f
#     return p
#
#
# def _sum(l):
#     s = 0
#     for t in l: s += t
#     return s
#
#
# def _diff(N, S):
#     return [e for e in N if e not in S]
#
#
# def prob_set(S, P, L):
#     n = len(P)
#     s = len(S)
#     T = _diff(range(n), S)
#     return L[s]*_prob(P[i] for i in S)*_prob(1-P[i] for i in T)
#
#
# def prob_family(F, P, L=None):
#     if L is None:
#         n = len(P)
#         L = [1]*(n+1)
#     return sum(prob_set(S, P, L) for S in F)
#
#
# def prob_elements(F, P, L):
#     n = len(P)
#     E = [0]*n
#     for S in F:
#         p = prob_set(S, P, L)
#         for e in S:
#             E[e] += p
#     return E
#
#
# def prob_family_elements(k, P, L=None):
#     n = len(P)
#     if k is None:
#         k = (0, n)
#     if isinstance(k, int):
#         k = (k, k)
#     if len(k) == 1:
#         k = (0, k[0])
#     if L is None:
#         l = [1]*(n+1)
#
#     N = list(range(n))
#     tot = prob_family(subsetsc(N, k[0], k[1]), P)
#     pel = prob_elements(subsetsc(N, k[0], k[1]), P)
#     return [pel[i]/tot for i in N]
#
# def prob_family_elements2(k, L, P):
#     n = len(P)
#     if k is None:
#         k = (0, n)
#     if isinstance(k, int):
#         k = (k, k)
#     if len(k) == 1:
#         k = (0, k[0])
#
#     N = list(range(n))
#     tot = prob_family(subsetsc(N, k[0], k[1]), P)
#     pel = prob_elements(subsetsc(N, k[0], k[1]), P)
#     return [pel[i]/tot for i in N]
#
#
# def uniform(n):
#     u = [random() for i in range(n)]
#     s = _sum(u)
#     return [p/s for p in u]
#
#
