import networkx as nx
import numpy as np
import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import PC, GES
from castle.algorithms import ANMNonlinear, ICALiNGAM, DirectLiNGAM
from castle.algorithms import Notears, NotearsNonlinear, GOLEM
from castle.common.priori_knowledge import PrioriKnowledge
from castle.common.independence_tests import hsic_test
from random import shuffle
import matplotlib.pyplot as plt


SEED = 42
np.random.seed(SEED)
COLORS = [
    '#00B0F0',
    '#FF0000',
    '#B0F000'
]


def reorder_array(data, permutation):
    n = len(permutation)
    shuffled = np.zeros_like(data)

    for i in range(n):
        k = permutation[i]
        shuffled[:, i] = data[:, k]
    return shuffled


def reorder_matrix(m, permutation, inverse=True):
    n = len(permutation)
    o = np.ones_like(m)

    if inverse:
        for r in range(n):
            for c in range(n):
                i = permutation[r]
                j = permutation[c]
                o[i, j] = m[r, c]
                # o[r, c] = m[i, j]
    else:
        for r in range(n):
            for c in range(n):
                i = permutation[r]
                j = permutation[c]
                # o[i, j] = m[r, c]
                o[r, c] = m[i, j]
    return o


def plot_dag(adj_matrix):
    g = nx.DiGraph(adj_matrix)

    plt.figure(figsize=(12, 8))
    nx.draw(
        G=g,
        node_color=COLORS[0],
        node_size=1200,
        arrowsize=17,
        with_labels=True,
        font_color='white',
        font_size=21,
        pos=nx.circular_layout(g)
    )
    plt.show()


def main():
    N = 10
    E = 17

    permutation = list(range(N))
    shuffle(permutation)
    print(permutation)

    # -----------------------------------------------------------------------
    # Check reorder
    # -----------------------------------------------------------------------
    m0 = np.array([i for i in range(N*N)]).reshape((N, N))

    # print(m0)
    m1 = reorder_matrix(m0, permutation, False)
    # print(m1)
    m2 = reorder_matrix(m1, permutation, True)
    # print(m2)
    assert (m0 == m2).all()

    # -----------------------------------------------------------------------
    # Data generation
    # -----------------------------------------------------------------------

    # Generate a scale-free adjacency matrix
    adj_matrix = DAG.scale_free(
        n_nodes=N,
        n_edges=E,
        seed=SEED
    )

    # Visualize the adjacency matrix
    plot_dag(adj_matrix)

    dataset = IIDSimulation(
        W=adj_matrix,
        n=10000,
        method='linear',
        sem_type='gauss'
    )

    # -----------------------------------------------------------------------
    # Original order
    # -----------------------------------------------------------------------

    pc = PC()
    # Fit the model
    pc.learn(dataset.X)

    pred_dag = pc.causal_matrix
    print("1)\n", pred_dag, pred_dag.sum().sum())

    plot_dag(pred_dag)

    # -----------------------------------------------------------------------
    # Shuffled order
    # -----------------------------------------------------------------------

    # shuffle the columns
    Xo = dataset.X
    Xr = reorder_array(Xo, permutation)

    pc = PC()
    # Fit the model
    pc.learn(Xr)

    pred_dag = pc.causal_matrix

    # reorder the adjacency matrix
    # pred_dag = reorder_matrix(pred_dag, permutation, True)
    # print("2)\n", pred_dag, pred_dag.sum().sum())
    # plot_dag(pred_dag)

    pred_dag = reorder_matrix(pred_dag, permutation, True)
    print("3)\n", pred_dag, pred_dag.sum().sum())
    plot_dag(pred_dag)


    pass


if __name__ == "__main__":
    main()

