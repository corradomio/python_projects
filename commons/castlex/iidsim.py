from typing import Union

from castle.datasets import simulator as sim
import numpy as np
import networkx as nx
import scipy


# Note: it is used a 'custom' implementation of
#
#       castle.datasets.IIDSimulation
#
# See the comment at the begin of the file './simulator.py'

class IIDSimulation:
    """
    Reimplementation of 'castle.datasets.IIDSimulation' with
    an interface more similar to 'scikit-learn':

        - constructor(method/sem_type)
        - fit(adjacency_matrix)
        - generate(n_of_samples)

    """
    def __init__(self, method='linear',  sem_type='gauss'):
        self.method = method
        self.sem_type = sem_type
        self.W = None

    def fit(self, W: Union[np.ndarray, nx.DiGraph, scipy.sparse.csr_array]):
        """
        Register the adjacency matrix. It can be

            1. a numpy array
            2. a networkx DiGraph
            3. a scipy 'csr_array'

        :param W: adjacency matrix
        """
        # note: B is a 'csr_array'. It can be converted in a numpy array with 'B.A'
        if isinstance(W, np.ndarray):
            self.W: np.ndarray = W
        elif isinstance(W, nx.DiGraph):
            G = W
            self.W: np.ndarray = nx.adjacency_matrix(G).toarray()
        elif isinstance(W, scipy.sparse.csr_array):
            self.W: np.ndarray = W.toarray()
        else:
            raise ValueError(f"Unsupported adjacency matrix of type {type(W)}")
        return self

    def generate(self, n=1000) -> np.ndarray:
        """
        Generate the dataset based on the adjacency matrix and method/sem_type
        specified in the constructor

        :param n: n of samples to generate
        :return: the dataset [n, order] (n of samples x graph order)
        """
        iids = sim.IIDSimulation(W=self.W, n=n, method=self.method, sem_type=self.sem_type)
        # B, X = iids.B, iids.X
        # return X
        return iids.X
# end
