import numpy as np
import numpy.random

N = 100


def randmat(n: int, symmetric: bool=True):
    m = np.random.rand(n,n)
    if symmetric:
        for i in range(n):
            for j in range(i+1, n):
                m[j,i] = m[i,j]
    return m



def main():
    m = randmat(N)
    pass


if __name__ == "__main__":
    main()


