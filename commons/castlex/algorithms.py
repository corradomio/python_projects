import castle.algorithms as algos


class Algo:
    def __init__(self):
        self.learner = None

    def fit(self, X) -> "Algo":
        ...

    def predict(self, X):
        ...

    def fit_predict(self, X):
        return self.fit(X).predict(X)
# end


class PC(Algo):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.PC(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class ANMNonlinear(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.ANMNonlinear(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class CORL(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.CORL(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class DAG_GNN(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.DAG_GNN(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class DirectLiNGAM(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.DirectLiNGAM(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class GAE(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.GAE(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class GES(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.GES(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class GOLEM(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.GOLEM(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class GraNDAG(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.GraNDAG(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class ICALiNGAM(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.ICALiNGAM(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns)
        return self.learner.causal_matrix
# end


class MCSL(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.MCSL(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class Notears(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.Notears(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class NotearsLowRank(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.NotearsLowRank(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class NotearsNonlinear(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.NotearsNonlinear(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class PNL(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.PNL(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class RL(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.RL(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


class TTPM(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.learner = algos.TTPM(*args, **kwargs)

    def fit(self, X):
        return self

    def predict(self, X, columns=None, **kwargs):
        self.learner.learn(X, columns=columns, **kwargs)
        return self.learner.causal_matrix
# end


