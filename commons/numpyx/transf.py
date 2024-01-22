#
# Numpy transformers
#


class Transformer:

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        ...

    def inverse_transform(self, X):
        ...

    def fit_transform(self, X):
        return self.fit(X).transform(X)

