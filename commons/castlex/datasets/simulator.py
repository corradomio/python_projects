from random import choice
from typing import Union, cast

from castle.datasets import simulator as sim
from scipy.special import expit as sigmoid
from itertools import combinations
import numpy as np
import networkx as nx
import scipy
import netx

def generate_quadratic_coef(random_zero=True):
    if random_zero and np.random.randint(low=0, high=2):
        return 0
    else:
        coef = np.random.uniform(low=0.5, high=1)
        if np.random.randint(low=0, high=2):
            coef *= -1
        return coef


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

    2025/01/13 Extension:
        support multiple methods and multiple sem_types
    """
    def __init__(self,
                 method: str|list[str]='linear',
                 sem_type: str|list[str]='gauss',
                 noise_scale: float=1.0):

        self.noise_scale = noise_scale
        self.methods = method if isinstance(method, list|tuple) else [method]
        self.sem_types = sem_type if isinstance(sem_type, list|tuple) else [sem_type]
        self.W = None

        assert isinstance(self.noise_scale, float)
        assert isinstance(self.methods, list)
        assert isinstance(self.sem_types, list)

    def fit(self, W: Union[np.ndarray, nx.DiGraph, scipy.sparse.csr_array]):
        """
        Register the adjacency matrix. It can be

            1. a numpy array
            2. a networkx DiGraph
            3. a scipy 'csr_array'

        :param W: adjacency matrix
        """
        # note: W is a 'csr_array'. It can be converted in a numpy array with 'B.A'
        if isinstance(W, np.ndarray):
            self.W: np.ndarray = W
        elif isinstance(W, nx.DiGraph):
            G = W
            self.W: np.ndarray = nx.adjacency_matrix(G).toarray()
        elif isinstance(W, netx.Graph) and cast(netx.Graph, W).is_dag():
            G = W
            self.W: np.ndarray = nx.adjacency_matrix(G).toarray()
        elif isinstance(W, scipy.sparse.csr_array):
            self.W: np.ndarray = W.toarray()
        else:
            raise ValueError(f"Unsupported adjacency matrix of type {type(W)}")
        return self

    def transform(self, n=1000) -> np.ndarray:
        return self.generate(n)

    def generate(self, n=1000) -> np.ndarray:
        """
        Generate the dataset based on the adjacency matrix and method/sem_type
        specified in the constructor.

        :param n: n of samples to generate
        :return: the dataset [n, order] (n of samples x graph order)
        """
        # Use the original implementation
        if (len(self.methods) == 1
                and self.methods[0] == "linear"
                and len(self.sem_types) == 1
                and self.sem_types[0] in ["gauss", "exp", "gumbel", "uniform", "logistic"]
        ):
            iidsim = sim.IIDSimulation(W=self.W, n=n, method="linear", sem_type=self.sem_types[0])
            X = iidsim.X
        elif (len(self.methods) == 1
              and len(self.sem_types) == 1
              and self.methods[0] in ["mlp", "mim", "gp", "gp-add", "quadratic"]
              and self.sem_types[0] == "gauss"):
            iidsim = sim.IIDSimulation(W=self.W, n=n, method="nonlinear", sem_type=self.methods[0])
            X = iidsim.X
        elif (len(self.methods) == 1
              and len(self.sem_types) == 1
              and self.sem_types[0] == "nonlinear"
              and self.sem_types[0] in ["mlp", "mim", "gp", "gp-add", "quadratic"]
        ):
            iidsim = sim.IIDSimulation(W=self.W, n=n, method="nonlinear", sem_type=self.sem_types[0])
            X = iidsim.X
        else:
            # new implementation
            X = self._generate_data(n)
        return X
    # end

    def _generate_data(self, n:int) -> np.ndarray:
        W = self.W
        noise_scale  = self.noise_scale

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx = nx.from_numpy_array(W, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')

        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = self._simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        return X
    # end

    def _simulate_single_equation(self, X: np.ndarray, w: np.ndarray, scale: float):
        # X[:, parents], W[parents, j], scale_vec[j]
        # X: dataset
        # W: adjacency matrix
        #
        n = X.shape[0]
        pa_size = X.shape[1]
        parents = list(range(pa_size))

        method = choice(self.methods)
        sem_type = choice(self.sem_types)

        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            # x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            # x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            # x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            # x = X @ w + z
        elif sem_type == 'logistic':
            z = np.zeros(n)
            # x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        else:
            raise ValueError('Unknown sem type. In a linear model, \
                                 the options are as follows: gauss, exp, \
                                 gumbel, uniform, logistic.')

        if method == 'linear':
            x = X @ w + z
        elif method == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0 + z
        elif method == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif method == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif method == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif method == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        elif method == "quadratic":
            if len(parents) == 0:
                eta = np.zeros([n])
            elif len(parents) == 1:
                eta = np.zeros([n])
                used_parents = set()
                p = parents[0]
                num_terms = 0

                # Linear term
                coef = generate_quadratic_coef(random_zero=False)
                if coef != 0:
                    eta += coef * X[:, p]
                    used_parents.add(p)
                    num_terms += 1

                # Squared term
                coef = generate_quadratic_coef(random_zero=False)
                if coef != 0:
                    eta += coef * np.square(X[:, p])
                    used_parents.add(p)
                    num_terms += 1

                if num_terms > 0:
                    eta /= num_terms    # Compute average

                if p not in used_parents:
                    w[p] = 0
            else:
                eta = np.zeros([n])
                used_parents = set()
                num_terms = 0

                for p in range(pa_size):
                    # Linear terms
                    coef = generate_quadratic_coef(random_zero=True)
                    if coef > 0:
                        eta += coef * X[:, p]
                        used_parents.add(p)
                        num_terms += 1

                    # Squared terms
                    coef = generate_quadratic_coef(random_zero=True)
                    if coef > 0:
                        eta += coef * np.square(X[:, p])
                        used_parents.add(p)
                        num_terms += 1

                # Cross terms
                for p1, p2 in combinations(parents, 2):
                    coef = generate_quadratic_coef(random_zero=True)
                    if coef > 0:
                        eta += coef * X[:, p1] * X[:, p2]
                        used_parents.add(p1)
                        used_parents.add(p2)
                        num_terms += 1

                if num_terms > 0:
                    eta /= num_terms    # Compute average

                # Remove parent if both coef is zero
                unused_parents = set(parents) - used_parents
                if p in unused_parents:
                    w[p] = 0
            # end
            x = eta + np.random.normal(scale=scale, size=n)
        else:
            raise ValueError('Unknown method. The options are as follows: mlp, mim,  gp, gp-add, or quadratic.')

        return x
# end
