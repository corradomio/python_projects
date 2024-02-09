from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from common import *


class ArrayBayesianOptimization(BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TargetFunction:
    """
    This class can be used as a simple function, because it implements
    the method __call__.
    The 'function' parameters is the list of D*M float used to generate
    the distilled dataset composed by D points with M dimensions with
    values in [0,1]
    """

    def __init__(self, X, y, D):
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

        def array_of(kwargs):
            # convert the list of parameters in a vector
            # the parameter's name is an integer in string form
            # used to specify the location in the array
            a = np.zeros(len(kwargs), dtype=float)
            for k in kwargs:
                a[int(k)] = kwargs[k]
            return a

        # just to avoid 'self.xxx'
        create_classifier = self.create_classifier
        X = self.X
        y = self.y
        D = self.D
        M = self.M
        GTC = self.GTC

        # convert the parameters in an array
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

        # compute the distilled classifier/dataset score
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

    plot_data("Data", X, y, bounds)

    # data dimensions
    N, M = X.shape
    # number of distilled points
    D = N//10

    # create the target function
    # it is used a class because the function
    # requires extra information, not only the
    # position of the distilled data points
    target_function = TargetFunction(X, y, D)

    # create the parameters for the BO
    # Note/1: the parameter's name can be not an integer, but it MUST be a string
    # Note/2: the parameter's name is an integer in string form. In this way it
    # is simple to convert the string into an integer and to use this value to
    # populate a numpy array
    bo_params = {
        f"{i:03}": (0., 1.)
        for i in range(D * M)
    }
    # create the BayesOpt
    # Note: used a custom class to add extra features to the
    # original implementation
    BO = ArrayBayesianOptimization(target_function, bo_params)

    # maximize the target function
    BO.maximize(init_points=2, n_iter=10, kappa=5)

    # retrieve the best results directly from TargetFunction
    accuracy = target_function.best_score
    model = target_function.best_model
    Xd = target_function.best_params
    yd = model.predict(Xd)

    # Some logs
    print("distilled accuracy:", accuracy)
    plot_data("Distilled", Xd, yd, bounds)

    # create the coreset
    Xcs, ycs = CoresetSelector(X, y).select(Xd)
    # create the distilled classifier based on the core set
    DCcs = target_function.create_classifier(Xcs, ycs)
    # apply the classifier on the original dataset
    yp = DCcs.predict(X)

    # compute the accuracy
    cs_accuracy = accuracy_score(y, yp)

    print("coreset accuracy:", cs_accuracy)
    plot_data("Coreset", Xcs, ycs, bounds)
    pass


if __name__ == "__main__":
    main()
