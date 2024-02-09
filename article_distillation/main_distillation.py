import numpy as np
import pandas as pd
import umap
import numpy.linalg as npl

from random import random
from numpy import ndarray
from path import Path as path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from skopt import gp_minimize


DATA_DIR = "proj_poly_3"
K = 10


def load_data(f: path) -> tuple[ndarray, ndarray, int]:
    print(f"... loading file")
    df = pd.read_csv(f)

    # df columns: y,x0,....,p00,...
    # y: target
    # xi: coordinates in low dimensions
    xcols = [c for c in df.columns if c.startswith("x")]
    # pj: coordinates in high dimension
    pcols = [c for c in df.columns if c.startswith("p")]

    y = df[['y']].to_numpy(dtype=int)
    X = df[pcols].to_numpy(dtype=float)

    return X, y, len(xcols)


def dimensional_reduction(X: ndarray, y: ndarray, ldim: int) -> ndarray:
    print(f"... dimensional_reduction: {ldim}")
    reducer = umap.UMAP(n_components=ldim)
    Xr = reducer.fit_transform(X, y)
    return Xr


class TargetFunction:
    """
    This class can be used as a simple function, because it implements
    the method __call__.
    The 'function' parameters is the list of D*M float used to generate
    the distilled dataset composed by D points with M dimensions with
    values in [0,1]
    """

    def __init__(self, X, y, D, maximize=True):
        self.X = X
        self.y = y
        self.D = D
        self.M = X.shape[1]
        # create the Ground Truth classifier
        self.GTC = self.create_classifier(X, y)
        # best results
        self.best_score = 0
        self.best_model = None
        self.best_params = None
        self.maximize = maximize
        pass

    def create_classifier(self, X, y):
        # create the classifier
        # classifier = RandomForestClassifier(n_estimators=16)
        classifier = DecisionTreeClassifier()
        # train the classifier using (X, y)
        classifier.fit(X, y)

        # just for log
        # yp = classifier.predict(X)
        # print("Accuracy:", accuracy_score(y, yp))
        return classifier

    def __call__(self, *args, **kwargs):
        # just to avoid 'self.xxx'
        create_classifier = self.create_classifier
        X = self.X
        y = self.y
        D = self.D
        M = self.M
        GTC = self.GTC

        # convert the parameters in an array
        x_ = np.array(args)

        # create Xd, the distilled dataset
        Xd = x_.reshape(D, M)
        # retrieve the labels for the distilled dataset
        yd = GTC.predict(Xd)

        # create & train the distilled classifier
        DC = create_classifier(Xd, yd)
        # use the distilled classifier to predict the labels of the
        # original dataset
        yp = DC.predict(X)

        # compute the distilled classifier/dataset score
        d_score = accuracy_score(y, yp)

        # save the best model
        if d_score > self.best_score:
            self.best_score = d_score
            self.best_model = DC
            self.best_params = Xd
        return d_score if self.maximize else (1-d_score)


class CoresetSelector:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def select(self, Xd):
        n = len(Xd)
        Xcs = np.zeros_like(Xd, dtype=self.X.dtype)
        ycs = np.zeros(n, dtype=self.y.dtype)
        for i in range(n):
            xcs, ccs = self._find_nearest(Xd[i])
            Xcs[i] = xcs
            ycs[i] = ccs[0]
        return Xcs, ycs

    def _find_nearest(self, d):
        X = self.X
        y = self.y
        n = len(X)

        best_point = X[0]
        best_class = y[0]
        best_dist = npl.norm(d-best_point)
        for i in range(n):
            x = X[i]
            dist = npl.norm(d-x)
            if dist < best_dist:
                best_dist = dist
                best_point = x
                best_class = y[i]
        # end
        return best_point, best_class
    # end


def find_distilled_points(Xr, y):
    print(f"... find_distilled_points")
    # ranges of the coordinates
    minr = Xr.min(axis=0)
    maxr = Xr.max(axis=0)
    diff = maxr - minr

    # N: n of elements, M: dimension
    N, M = Xr.shape
    # Nr: number of distilled points
    Nd = N//K

    # create the target function
    # it is used a class because the function
    # requires extra information, not only the
    # position of the distilled data points
    target_function = TargetFunction(Xr, y, Nd, maximize=False)

    # create the parameters for the BayesOpt
    # Note/1: the parameter's name can be not an integer, but it MUST be a string
    # Note/2: the parameter's name is an integer in string form. In this way it
    # is simple to convert the string into an integer and to use this value to
    # populate a numpy array
    #
    # the number of parameters is Nd (n of distilled points) times M (the points dimension)
    # initialize 'bounds', the valid ranges for each parameter
    gp_bounds = [
        (minr[i % M], maxr[i % M])
        for i in range(Nd * M)
    ]
    # initialize 'x0', the initial value for each parameter
    gp_x0 = [minr[i % M] + diff[i % M]*random() for i in range(Nd * M)]

    # create the BayesOpt
    res = gp_minimize(target_function, gp_bounds,
                      x0=gp_x0,
                      acq_func="LCB",
                      n_calls=15,
                      n_random_starts=3,
                      n_points=1000,
                      random_state=777,
                      verbose=False)

    # retrieve the best results directly from TargetFunction
    accuracy = target_function.best_score
    model = target_function.best_model
    Xd = target_function.best_params
    yd = model.predict(Xd)

    # Some logs
    print("... ... distilled accuracy:", accuracy)
    # plot_data("Distilled", Xd, yd, bounds)

    return model, Xd, yd, target_function


def find_coreset(Xr, y, Xd, target_function):
    print(f"... find_coreset")
    # create the coreset
    Xcs, ycs = CoresetSelector(Xr, y).select(Xd)
    # create the distilled classifier based on the core set
    DC = target_function.create_classifier(Xcs, ycs)

    # apply the classifier on the original dataset
    yp = DC.predict(Xr)
    # compute the accuracy
    cs_accuracy = accuracy_score(y, yp)
    print("... ... coreset accuracy:", cs_accuracy)
    return Xcs, ycs


def process_file(f: path):
    print(f"Processing {f.stem} ...")
    # load data
    X, y, ldim = load_data(f)
    # dimensionality reduction
    Xr = dimensional_reduction(X, y, ldim)
    # find the best model and distilled points
    model, Xd, yd, target_function = find_distilled_points(Xr, y)
    # find the coreset
    Xcs, yc = find_coreset(Xr, y, Xd, target_function)
    return


def main():
    data_dir = path(DATA_DIR)
    for f in data_dir.files("*csv"):
        process_file(f)



if __name__ == "__main__":
    main()
