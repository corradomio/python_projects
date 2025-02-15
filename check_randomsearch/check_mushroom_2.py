from datetime import datetime
from random import randrange, shuffle
from typing import Optional

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import pandasx as pdx
import stdlib.jsonx as jsx
from pandasx.preprocessing import BinHotEncoder
from sklearnx.model_selection import RandomizedSearchCV, Const, BayesSearchCV
from stdlib.tprint import tprint, delta_time


def choices_n(n, k):
    seq = list(range(n))
    shuffle(seq)
    return seq[:k]


class GroundTruthInfo:

    def __init__(self, X: DataFrame, y: DataFrame):
        self.X: DataFrame = X
        self.y: DataFrame = y
        self.xenc = BinHotEncoder().fit(X)
        self.yenc = BinHotEncoder().fit(y)
        
        self.C = self.create_classifier(X, y)
        self.columns = list(X.columns)
        pass

    def compose(self, kwparams):
        # skip 'GT'
        D = len(kwparams)-1
        selected = [
            kwparams[f'xi_{i:02}'] for i in range(D)
        ]
        M = self.X.iloc[sorted(selected)]
        return M

    def create_classifier(self, X, y):
        Xenc = self.xenc.transform(X)
        yenc = self.yenc.transform(y)
        c = DecisionTreeClassifier().fit(Xenc, yenc)
        return c

    def predict(self, Xd):
        Xe = self.xenc.transform(Xd)
        ye = self.C.predict(Xe)
        ye = DataFrame(data=ye, columns=self.y.columns, index=Xd.index)
        yp = self.yenc.inverse_transform(ye)
        return yp


class Parameters:
    def __init__(self, D: int, GT: GroundTruthInfo):
        self.D = D
        self.GT = GT
        n = len(self.GT.X)

    def bounds(self) -> list:
        """
        :param n: number of continuous values to consider
        """
        D = self.D
        n = len(self.GT.X)
        bounds = [list(range(n)) for i in range(D)]
        return bounds

    def named_bounds(self, Xy: bool = False) -> dict:
        D = self.D
        n = len(self.GT.X)
        named_bounds = {
            f'xi_{i:02}': list(range(n)) for i in range(D)
        }
        if Xy:
            named_bounds['GT'] = [Const(self.GT)]
        return named_bounds

    def values(self):
        n = len(self.GT.X)
        values = choices_n(n, self.D)
        return values

    def named_values(self, Xy: bool = False):
        D = self.D
        n = len(self.GT.X)
        values = choices_n(n, D)
        named_values = {
            f'xi_{i:02}': values[i] for i in range(D)
        }
        if Xy:
            named_values['GT'] = Const(self.GT)
        return named_values


class BestScores:
    def __init__(self, D: int, X: Optional[DataFrame], y: Optional[DataFrame]):
        self.D = D  # n of distilled points
        N, M = X.shape if X is not None else 0, 0
        T = 0 if y is None else 1 if len(y.shape) == 1 else y.shape[1]
        self.N = N  # n of points in dataset
        self.M = M  # n of features
        self.T = T  # n of targets
        self.iter = 0
        self.score_history = []
        self.best_score_history = []
        self.best_iter = 0
        self.best_score = -1
        self.best_params = None
        self.best_model = None
        self.start_time = datetime.now()
        self.done_time = datetime.now()

    def update(self, score, params, model):
        self.score_history.append(score)
        if score > self.best_score:
            self.best_iter = self.iter
            self.best_score = score
            self.best_params = params
            self.best_model = model
            self.best_score_history.append({'iter': self.iter, 'score': score})
            tprint(f"   {self.iter}: {score:.3} ***")
        else:
            tprint(f"   {self.iter}: {score:.3}")
        self.iter += 1

    def save(self, fname):
        self.done_time = datetime.now()

        D = self.D
        date_ext = self.start_time.strftime("%Y%m%d.%H%M%S")
        cname = f"{fname}-{D}-{date_ext}.csv"
        jname = f"{fname}-{D}-{date_ext}.json"

        df = pd.concat(self.best_params, axis=1)
        pdx.save(df, cname, index=False)
        jsx.save({
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "done_time": self.done_time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": delta_time(self.start_time, self.done_time),
            "synthetic_points": True,

            "n_iter": len(self.score_history),
            "n_distilled_points": self.D,
            "n_features": self.M,
            "n_targets": self.T,

            "classifier": self.best_model.__class__.__name__,

            "best_score": {"iter": self.best_iter, "score": self.best_score},
            "score_history": self.score_history,
            "best_score_history": self.best_score_history
        }, jname)
        pass


BEST_SCORES: BestScores = BestScores(0, None, None)


class DistilledModel:
    _estimator_type = "classifier"

    def __init__(self, **kwparams):
        self.kwparams = None
        self.Xd = None
        self.yd = None
        self.GT = None
        self.DC = None
        self.set_params(**kwparams)

    def get_params(self, deep=True):
        return self.kwparams

    def set_params(self, **kwparams):
        self.kwparams = kwparams
        self.GT: GroundTruthInfo = kwparams['GT'].value
        self.Xd: DataFrame = self.GT.compose(kwparams)
        self.yd = self.GT.predict(self.Xd)

        self.DC = self.GT.create_classifier(self.Xd, self.yd)
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        Xe = self.GT.xenc.transform(X)
        ye = self.DC.predict(Xe)
        ye = DataFrame(data=ye, columns=self.GT.y.columns, index=X.index)
        ypred = self.GT.yenc.find_nearest(ye)
        return ypred

    def score(self, X, y_true):
        y_pred = self.predict(X)
        score = accuracy_score(y_true, y_pred)

        global BEST_SCORES
        BEST_SCORES.update(score, (self.Xd, self.yd), self.DC)

        return score
# end


def load_data():
    df = pdx.read_data(r"D:\Projects.github\article_projects\article_distillation\data_uci\mushroom\mushroom.csv",
                       categorical=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                                    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                                    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
                                    ]
                       )

    X, y = pdx.xy_split(df, target='target')
    # X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=.25)
    # return X_train, X_test, y_train, y_test
    return X, y


def main():
    global BEST_SCORES

    X, y = load_data()
    D = 100

    GT = GroundTruthInfo(X, y)
    BEST_SCORES = BestScores(D, X, y)
    parameters = Parameters(D, GT)

    kwparams = parameters.named_values(Xy=True)
    kwbounds = parameters.named_bounds(Xy=True)

    estimator = DistilledModel(**kwparams)

    # opt = RandomizedSearchCV(
    #     estimator=estimator,
    #     param_distributions=kwbounds,
    #     scoring=lambda estimator, X, y: estimator.score(X, y),
    #     n_iter=10,
    #     n_jobs=1,
    #     cv=2
    # )

    opt = BayesSearchCV(
        estimator=estimator,
        param_distributions=kwbounds,
        scoring=lambda estimator, X, y: estimator.score(X, y),
        n_points=1,
        n_iter=15,
        n_jobs=1,
        verbose=1,
        cv=2,
        # pre_dispatch=1
        optimizer_kwargs=dict(
            acq_optimizer_kwargs=dict(
                n_points=10,
                n_restarts_optimizer=5,
                n_jobs=1
            ),
            acq_func_kwargs=dict(

            )
        )
    )

    opt.fit(X, y)

    BEST_SCORES.save('pred/mushroom')
    pass


if __name__ == "__main__":
    main()
