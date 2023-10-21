from . import simulator as sim
import numpy as np
import networkx as nx
import scipy


class IIDSimulation:
    def __init__(self, method='linear',  sem_type='gauss'):
        self.method = method
        self.sem_type = sem_type
        self.W = None

    def fit(self, W):
        """
        Register the adjacency matrix. It can be

            1. a numpy array
            2. a networkx DiGraph
            3. a scipy 'csr_array'

        :param W: adjacency matrix
        """
        # note: B is a 'csr_array'. It can be converted in a numpy array with 'B.A'
        if isinstance(W, np.ndarray):
            self.W = W
        elif isinstance(W, nx.DiGraph):
            G = W
            self.W = nx.adjacency_matrix(G).toarray()
        elif isinstance(W, scipy.sparse.csr_array):
            self.W = W.toarray()
        else:
            raise ValueError(f"Unsupported adjacency matrix of type {type(W)}")
        return self

    def generate(self, n=2000):
        """
        Generate the dataset based on the adjacency matrix d used in the constructor

        :param n: n of samples to generate
        :return:the dataset
        """
        iids = sim.IIDSimulation(W=self.W, n=n, method=self.method, sem_type=self.sem_type)
        B, X = iids.B, iids.X
        return X
