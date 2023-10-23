import networkx as nx
import matplotlib.pyplot as plt
import netx as nxx
from causalx import IIDSimulation
from causalx.algorithms import PC


def gen():
    G = nxx.random_dag(10, 20)

    nxx.draw(G)
    plt.show()

    M = nx.adjacency_matrix(G).A
    # X = IIDSimulation(method='nonlinear', sem_type='quadratic').fit(M).generate(2000)
    X = IIDSimulation(method='linear', sem_type='gauss').fit(M).generate(2000)

    pc = PC()
    C = pc.fit_predict(X)

    nxx.draw(C)
    plt.show()


def main():
    print("Hello World")
    gen()

    pass


if __name__ == "__main__":
    main()
