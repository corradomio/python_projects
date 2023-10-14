import castle.datasets as cds
import numpy as np
import networkx as nx


class IIDSimulation:
    def __init__(self, method='linear',  sem_type='gauss'):
        self.method = method
        self.sem_type = sem_type
        self.W = None

    def fit(self, W):
        if isinstance(W, np.ndarray):
            self.W = W
        elif isinstance(W, nx.DiGraph):
            G = W
            self.W = nx.adjacency_matrix(G).toarray()
        else:
            raise ValueError(f"Unsupported adjacency matrix of type {type(W)}")
        return self

    def generate(self, n=2000):
        """

        :param n:
        :return: true causal matrix, dataset
        """
        iids = cds.IIDSimulation(W=self.W, n=n, method=self.method, sem_type=self.sem_type)
        # note: B is a 'csr_array'. It can be converted in a numpy array with 'B.A'
        B, X = iids.B, iids.X
        return X
