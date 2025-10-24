import numpy as np
import numpyx as npx
import netx
from stdlib.imathx import ilog2up


def main():
    N = 10
    I = npx.identity(N, dtype=int)
    G = netx.random_dag(N, 3*N//2)
    A = netx.adjacency_matrix(G)

    netx.draw(G)
    netx.show()

    p = ilog2up(N)
    P = npx.power2(A+I, p)
    print(A)
    print(P)





if __name__ == "__main__":
    main()
