from datetime import datetime
from random import shuffle

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from umap import UMAP

import pandasx as pdx
import stdlib.jsonx as jsx
from sklearnx.model_selection import RandomizedSearchCV, BayesSearchCV, Const
from stdlib.tprint import tprint, delta_time


def choices_n(n, k):
    seq = list(range(n))
    shuffle(seq)
    return seq[:k]


class GroundTruthInfo:

    def __init__(self, X: ndarray, y: ndarray):
        self.X: ndarray = X
        self.y: ndarray = y

        self.C = self.create_classifier(X, y)
        pass

    def compose(self, kwparams):
        # skip 'GT'
        m = self.X.shape[1]
        D = len(kwparams) // m
        data = []
        for i in range(D):
            di = []
            for j in range(m):
                di.append(kwparams[f'x_{i:02}_{j:02}'])
            data.append(di)
        # end
        M = np.array(data)
        return M

    def create_classifier(self, X, y):
        c = DecisionTreeClassifier().fit(X, y)
        return c

    def predict(self, Xd):
        yp = self.C.predict(Xd)
        return yp


class Parameters:
    def __init__(self, D: int, GT: GroundTruthInfo, XB):
        self.D = D
        self.GT = GT
        self.XB = XB

    def named_bounds(self, Xy: bool = False, n: int = 0) -> dict:
        D = self.D
        m = self.GT.X.shape[1]
        named_bounds = {}
        for i in range(D):
            for j in range(m):
                named_bounds[f'x_{i:02}_{j:02}'] = self.XB[j].bounds(n)
        if Xy:
            named_bounds['GT'] = [Const(self.GT)]
        return named_bounds

    def named_values(self, Xy: bool = False):
        D = self.D
        m = self.GT.X.shape[1]
        named_values = {}
        for i in range(D):
            for j in range(m):
                named_values[f'x_{i:02}_{j:02}'] = self.XB[j].random()
        if Xy:
            named_values['GT'] = Const(self.GT)
        return named_values


class BestScores:
    def __init__(self, D: int, X: ndarray, y: ndarray):
        self.D = D  # n of distilled points
        N, M = X.shape if X is not None else (0, 0)
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

    def save(self, fname, Xy=None, method='generic', coreset_score=0.):
        self.done_time = datetime.now()

        D = self.D
        date_ext = self.start_time.strftime("%Y%m%d.%H%M%S")
        cname = f"{fname}-{D}-{date_ext}.csv"
        jname = f"{fname}-{D}-{date_ext}.json"

        df = pd.concat(Xy, axis=1)
        pdx.save(df, cname, index=False)
        jsx.save({
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "done_time": self.done_time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": delta_time(self.start_time, self.done_time),
            "synthetic_points": True,

            "n_iter": len(self.score_history),
            "n_distilled_points": self.D,
            "n_generated_points": len(Xy[0]),
            "n_points": self.N,
            "n_features": self.M,
            "n_targets": self.T,
            "method": method,
            "classifier": self.best_model.__class__.__name__,

            "best_score": {"iter": self.best_iter, "score": self.best_score},
            "score_history": self.score_history,
            "best_score_history": self.best_score_history,
            "coreset_score": coreset_score
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
        self.Xd: ndarray = self.GT.compose(kwparams)
        self.yd = self.GT.predict(self.Xd)

        self.DC = self.GT.create_classifier(self.Xd, self.yd)
        return self

    def fit(self, X, y):
        return self

    def predict(self, Xe):
        yp = self.DC.predict(Xe)
        return yp.reshape((-1, 1))

    def score(self, X, y_true):
        y_pred = self.predict(X)
        score = accuracy_score(y_true, y_pred)
        # score = mean_squared_error(y_true, y_pred)

        global BEST_SCORES
        BEST_SCORES.update(score, (self.Xd, self.yd), self.DC)

        return score


# def load_data():
#     df = pdx.read_data(r"D:\Projects.github\article_projects\article_distillation\data_uci\mushroom\mushroom.csv",
#                        onehot=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#                                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
#                                'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
#                                'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
#                                'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
#                                ]
#                        )
#
#     Xy = pdx.xy_split(df, target='target')
#     X: DataFrame = Xy[0]
#     y: DataFrame = Xy[1]
#
#     dr = UMAP(n_components=10)
#     Xr = dr.fit(X, y).transform(X)
#     yr = y.to_numpy()
#
#     return Xr, yr, dr


class LoadData:
    def __init__(self, n_components=7):
        self.df = None
        self.dr = UMAP(n_components=n_components)

        self.Xo = None
        self.yo = None
        self.Xt = None
        self.yt = None
        self.Xr = None
        self.yr = None

        self.xenc = None
        self.yenc = None
        pass

    def load_data(self):
        print("Loading data...")
        df = pdx.read_data(
            r"D:\Projects.github\article_projects\article_distillation\data_uci\mushroom\mushroom.csv",
            # onehot=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            #         'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
            #         'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
            #         'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
            #         'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
            #         ]
            )

        self.xenc = pdx.OneHotEncoder([
            'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
            'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
            'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
        ])
        self.yenc = pdx.OneHotEncoder('target')

        print("... split")
        X, y = pdx.xy_split(df, target='target')
        self.Xo = X
        self.yo = y

        print("... transform")
        Xt = self.xenc.fit_transform(X)
        yt = self.yenc.fit_transform(y)
        self.Xt = Xt
        self.yt = yt

        print("... reduce")
        Xr = self.dr.fit(Xt, yt).transform(Xt)
        yr = yt.to_numpy()
        self.Xr = Xr
        self.yr = yr

        print("done")
        return Xr, yr

    def original_data(self):
        return self.Xr, self.yr

    def transform(self, X, y):
        Xt = self.xenc.transform(X)
        yt = self.yenc.transform(y)

        Xr = self.dr.transform(Xt)
        yr = yt.to_numpy()
        return Xr, yr

    def find_nearest(self, Xy):
        Xr, yr = Xy

        # too slow
        # Xt = self.dr.inverse_transform(Xr)
        # yt = yr

        Xi = []
        yi = []
        ii = []
        n = len(Xr)
        for i in range(n):
            xri = Xr[i]
            idx = self._find_nearest(xri)
            if idx in ii:
                continue
            Xi.append(self.Xo.iloc[idx])
            yi.append(self.yo.iloc[idx])
            ii.append(idx)

        Xi = pd.DataFrame(data=Xi, columns=self.Xo.columns, index=ii)
        yi = pd.DataFrame(data=yi, columns=self.yo.columns, index=ii)
        return Xi, yi

    def _find_nearest(self, xri) -> int:
        n = self.Xr.shape[0]
        min_dist = float('inf')
        selected: int = 0
        for i in range(n):
            dist = np.linalg.norm(self.Xr[i] - xri)
            if dist < min_dist:
                min_dist = dist
                selected = i
        return selected
# end


def main():
    global BEST_SCORES

    ld = LoadData()
    X, y = ld.load_data()

    X_bounds = pdx.columns_range(X)

    D = 100

    GT = GroundTruthInfo(X, y)
    BEST_SCORES = BestScores(D, X, y)
    parameters = Parameters(D, GT, X_bounds)

    # -----

    method = 'RandomizedSearchCV'
    kwparams = parameters.named_values(Xy=True)
    kwbounds = parameters.named_bounds(Xy=True, n=10)

    estimator = DistilledModel(**kwparams)

    opt = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=kwbounds,
        scoring=lambda estimator, X, y: estimator.score(X, y),
        n_iter=10,
        n_jobs=1,
        cv=5
    )

    # -----

    # method = 'BayesSearchCV'
    # kwparams = parameters.named_values(Xy=True)
    # kwbounds = parameters.named_bounds(Xy=True)
    #
    # estimator = DistilledModel(**kwparams)
    #
    # opt = BayesSearchCV(
    #     estimator=estimator,
    #     param_distributions=kwbounds,
    #     scoring=lambda estimator, X, y: estimator.score(X, y),
    #     n_points=1,
    #     n_iter=30,
    #     n_jobs=1,
    #     verbose=1,
    #     cv=5,
    #     # pre_dispatch=1
    #     optimizer_kwargs=dict(
    #         acq_optimizer_kwargs=dict(
    #             n_points=10,
    #             n_restarts_optimizer=5,
    #             n_jobs=1
    #         ),
    #         acq_func_kwargs=dict(
    #
    #         )
    #     )
    # )

    # -----

    opt.fit(X, y)

    # X,y distilled
    Xd, yd = ld.find_nearest(BEST_SCORES.best_params)
    # X,y distilled in the small dimension
    Xdr, ydr = ld.transform(Xd, yd)

    # create the distilled classifier
    DC = GT.create_classifier(Xdr, ydr)

    # original dataset in the small dimension
    Xr, yr = ld.original_data()

    yp = DC.predict(Xr)
    coreset_score = accuracy_score(yr, yp)

    BEST_SCORES.save('pred3/mushroom', (Xd, yd), method, coreset_score)
    pass


if __name__ == "__main__":
    main()
