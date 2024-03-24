from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import pandasx as pdx
import stdlib.jsonx as jsx
from pandasx.preprocessing import BinHotEncoder
from sklearnx.model_selection import RandomizedSearchCV, BayesOptSearchCV, Const
from stdlib.tprint import tprint, delta_time
from datetime import datetime


class GroundTruthInfo:

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X
        self.y = y
        self.xenc = BinHotEncoder().fit(X)
        self.yenc = BinHotEncoder().fit(y)
        self.C = self.create_classifier(X, y)
        self.columns = list(X.columns)
        pass

    def compose(self, kwvalues):
        n = len(kwvalues)//len(self.columns)
        M = []
        for i in range(n):
            mlist = []
            for col in self.columns:
                name = col.replace("-", "_")
                param = f"{name}_{i:02}"
                mlist.append(kwvalues[param])
            M.append(mlist)
        # end
        M = pd.DataFrame(data=M, columns=self.columns)
        return M

    def create_classifier(self, X, y):
        Xenc = self.xenc.transform(X)
        yenc = self.yenc.transform(y)
        c = DecisionTreeClassifier().fit(Xenc, yenc)
        return c

    def predict(self, Xd):
        Xe = self.xenc.transform(Xd)
        ye = self.C.predict(Xe)
        ye = pd.DataFrame(data=ye, columns=self.y.columns)
        yp = self.yenc.inverse_transform(ye)
        return yp


class Parameters:
    def __init__(self, D: int, columns_range: dict, GT: GroundTruthInfo):
        self.D = D
        # make the columns order 'consistent'
        self.columns = list(columns_range.keys())
        # categorical values
        self.column_ranges = columns_range
        self.GT = GT

    # def bounds(self, n: Optional[int] = None) -> list:
    #     """
    #     :param n: number of continuous values to consider
    #     """
    #     columns_range = self.column_ranges
    #     D = self.D
    #     bounds = [columns_range[col].bounds(n) for i in range(D) for col in self.columns]
    #     return bounds

    def named_bounds(self, Xy: bool = False, n: Optional[int] = None) -> dict:
        named_bounds = {}
        columns_range = self.column_ranges
        D = self.D
        for i in range(D):
            for col in self.columns:
                name = col.replace("-", "_")
                param = f"{name}_{i:02}"
                named_bounds[param] = columns_range[col].bounds(n)
            # end
        # end
        if Xy:
            named_bounds['GT'] = [Const(self.GT)]
        return named_bounds

    def values(self):
        columns_range = self.column_ranges
        D = self.D
        values = [columns_range[col].random() for i in range(D) for col in self.columns]
        return values

    def named_values(self, Xy: bool = False):
        named_values = {}
        columns_range = self.column_ranges
        D = self.D
        for i in range(D):
            for col in self.columns:
                name = col.replace("-", "_")
                param = f"{name}_{i:02}"
                named_values[param] = columns_range[col].random()
            # end
        # end
        if Xy:
            named_values['GT'] = Const(self.GT)
        return named_values


class BestScores:
    def __init__(self, D: int, X: Optional[pd.DataFrame], y: Optional[pd.DataFrame]):
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
        self.kwkeys = list(kwparams.keys())
        self.kwparams = None
        self.GT = None
        self.DC = None
        self.Xd = None
        self.yd = None
        self.set_params(**kwparams)

    def get_params(self, deep=True):
        return self.kwparams

    def set_params(self, **kwparams):
        self.kwparams = kwparams
        self.GT = kwparams['GT'].value
        self.Xd = self.GT.compose(kwparams)
        self.yd = self.GT.predict(self.Xd)

        self.DC = self.GT.create_classifier(self.Xd, self.yd)
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        Xe = self.GT.xenc.transform(X)
        ye = self.DC.predict(Xe)
        ye = pd.DataFrame(data=ye, columns=self.GT.y.columns, index=X.index)
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
    df = df.iloc[0:100]

    X, y = pdx.xy_split(df, target='target')
    # X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=.25)
    # return X_train, X_test, y_train, y_test
    return X, y


def main():
    global BEST_SCORES

    X, y = load_data()
    D = 100

    GT = GroundTruthInfo(X, y)
    columns_range = pdx.columns_range(X)

    BEST_SCORES = BestScores(D, X, y)
    parameters = Parameters(D, columns_range, GT)

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

    opt = BayesOptSearchCV(
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
