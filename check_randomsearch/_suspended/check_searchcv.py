import numpy as np
from random import choice
from sklearnx.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression


class NoneEstimator:

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.fc = 0
        self.pc = 0

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs
        return self

    def fit(self, X, y, **fit_params):
        self.fc += 1
        print("self.fit", self.fc)

        self.X = X
        self.y = y
        return self

    def predict(self, X, **predict_params):
        self.pc += 1
        print("self.predict", self.pc)
        return self.y

    def score(self, X, y, **score_params):
        self.pc += 1
        print("self.score", self.pc)
        return 1.


def generate_params_values(n_params=2000, n_values=4):
    params = {}
    values = [i for i in range(n_values)]
    for i in range(n_params):
        pname = f"p{i:03}"
        params[pname] = choice(values)
    return params


def generate_params_grid(n_params=2000, n_values=4):
    params = {}
    values = [i for i in range(n_values)]
    for i in range(n_params):
        pname = f"p{i:03}"
        params[pname] = values
    return params


def generate_Xy(N=100, M=5):
    X = [[i for i in range(M)] for j in range(N)]
    y = [j for j in range(N)]

    return np.array(X), np.array(y)


def main():
    params = generate_params_values(2000, 4)

    estimator = NoneEstimator(**params)

    params_grid = generate_params_grid(4, 4)

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params_grid,
        scoring=lambda estimator, X, y: estimator.score(X, y),
        n_iter=10,
        n_jobs=1,
        # cv=2
    )

    X, y = generate_Xy()

    rs.fit(X, y)

    pass


if __name__ == "__main__":
    main()