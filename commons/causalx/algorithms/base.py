class Algo:

    def fit(self, X) -> "Algo":
        ...

    def predict(self, X):
        ...

    def fit_predict(self, X):
        return self.fit(X).predict(X)
