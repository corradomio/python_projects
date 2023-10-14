import castle.algorithms as algos
from .base import Algo


class PC(Algo):
    def __init__(self, *args, **kwargs):
        self.pc = algos.PC(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.pc.learn(X, columns=columns, **kwargs)
        return self.pc.causal_matrix
