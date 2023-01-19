import numpy as np
import numpy.linalg as la
from mosek.fusion import *
from pprint import pprint


# points
S = 2       # space
N = 10      # n of data points
P = np.random.rand(N, S)

# centroids
card = [3, 3, 4]
K = len(card)
C = np.random.rand(K, S)

# distances point->centroid
D = (np.square(np.array([
    la.norm(P-C[0], axis=1),
    la.norm(P-C[1], axis=1),
    la.norm(P-C[2], axis=1),
])).T).tolist()

# euclidean distances [point,centroid]


with Model('jianyi') as M:
    T = M.variable('T', [N, K], Domain.integral(Domain.binary()))
    d = Matrix.dense(D)
    c = Matrix.dense(K, 1, card)
    U = M.variable('U', [K, K], Domain.integral(Domain.binary()))

    M.objective("obj", ObjectiveSense.Minimixe, Expr.sum(Expr.mulElm(d, T)))

    M.constraint(Expr.sum(T, 0), Domain.equalsTo(1))
    M.constraint(Expr.sum(U, 0), Domain.equalsTo(1))
    M.constraint(Expr.sum(U, 1), Domain.equalsTo(1))

    M.constraint(Expr.sub(Expr.sum(T, 1), Expr.sum(Expr.mul())))

    pass
