import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import json
from bayes_opt import BayesianOptimization


def plot_bounds(bounds):
    nx = len(bounds[0])
    for i in range(1, nx-1):
        bx = bounds[0][i]
        plt.plot((bx, bx), (0, 1), c='gray')

    ny = len(bounds[1])
    for i in range(1, ny-1):
        by = bounds[1][i]
        plt.plot((0, 1), (by, by), c='gray')


def plot_data(X, y, Xd, yd, bounds):
    plt.clf()

    X_blu = X[y == 0]
    X_red = X[y == 1]

    plt.scatter(X_blu[:, 0], X_blu[:, 1], c='blue', s=5)
    plt.scatter(X_red[:, 0], X_red[:, 1], c='red', s=5)
    plot_bounds(bounds)

    plt.savefig("Data.png", dpi=300)
    plt.clf()

    Xd_blu = Xd[yd == 0]
    Xd_red = Xd[yd == 1]

    plt.scatter(Xd_blu[:, 0], Xd_blu[:, 1], c='blue', s=20)
    plt.scatter(Xd_red[:, 0], Xd_red[:, 1], c='red', s=20)
    plot_bounds(bounds)

    plt.savefig("Distilled.png", dpi=300)


class ArrayBayesianOptimization(BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def best_results(self) -> tuple[float, np.ndarray]:
    #     best_results = super().max
    #     return best_results['target'], array_of(best_results['params'])


def load_data():
    df = pd.read_csv("Xy.csv")
    X = df[["x1", "x2"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    return X, y


def load_bounds():
    with open("Xy_bounds.json") as fp:
        return json.load(fp)


class TargetFunction:

    def __init__(self, X, y, D):
        self.X = X
        self.y = y
        self.GTC = self.create_classifier(X, y)
        self.D = D
        self.M = X.shape[1]

        self.best_score = 0
        self.best_model = None
        self.best_params = None
        pass

    def create_classifier(self, X, y):
        # classifier = RandomForestClassifier(n_estimators=16)
        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)

        yp = classifier.predict(X)
        print("Accuracy:", accuracy_score(y, yp))
        return classifier

    def __call__(self, *args, **kwargs):

        def array_of(kwargs):
            a = np.zeros(len(kwargs), dtype=float)
            for k in kwargs:
                a[int(k)] = kwargs[k]
            return a

        # just to avoid 'self.xxx'
        create_classifier = self.create_classifier
        D = self.D
        M = self.M
        GTC = self.GTC
        X = self.X
        y = self.y

        # convert the parameters in a numpy array
        x_ = array_of(kwargs)

        # create Xd, the distilled dataset
        Xd = x_.reshape(D, M)
        # retrieve the labels for the distilled dataset
        yd = GTC.predict(Xd)

        # create & train the distilled classifier
        DC = create_classifier(Xd, yd)
        # use the distilled classifier to predict the labels of the
        # original dataset
        yp = DC.predict(X)

        # compute the distilled dataset score
        d_score = accuracy_score(y, yp)

        # save the best model
        if d_score > self.best_score:
            self.best_score = d_score
            self.best_model = DC
            self.best_params = Xd
        return d_score


def main():
    # load the data
    bounds = load_bounds()
    X, y = load_data()
    N, M = X.shape
    D = N//10

    # create the target_function
    target_function = TargetFunction(X, y, D)

    # create the parameters for the BO
    bo_params = {
        f"{i:03}": (0., 1.)
        for i in range(D * M)
    }
    # create the BayesOpt
    BO = ArrayBayesianOptimization(target_function, bo_params)

    # maximize the target function
    BO.maximize(init_points=2, n_iter=10, kappa=5)

    # retrieve the best results directly from TargetFunction

    accuracy = target_function.best_score
    model = target_function.best_model
    Xd = target_function.best_params
    yd = model.predict(Xd)

    print("best accuracy:", accuracy)

    plot_data(X, y, Xd, yd, bounds)
    pass


if __name__ == "__main__":
    main()

    # # generate the first distilled dataset
    # Xd = generate_random_distilled_points(D, M)
    #
    # # train the ground truth classifier
    # GTC = create_classifier(X, y)
    #
    # end = False
    # while not end:
    #     # assign the labels to the distilled points
    #     yd = GTC.predict(Xd)
    #     # create the distilled classifier
    #     DC = create_classifier(Xd, yd)
    #     # predict the labels
    #     yp = DC.predict(X)
    #
    #     d_score = accuracy_score(y, yp)
    #
    #     bayesian_point = (Xd, d_score)
