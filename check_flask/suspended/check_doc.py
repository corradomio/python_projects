from sktime.forecasting.naive import NaiveForecaster

from utils import import_from
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


def main():
    c = NaiveForecaster
    clazz = import_from('sktime.forecasting.exp_smoothing.ExponentialSmoothing')
    print(clazz)
    print(clazz.__init__)

    print("Class doc")
    print(clazz.__doc__)
    print("Method doc")
    print(clazz.__init__.__doc__)
    print("Method kwargs")
    print(clazz.__init__.__kwdefaults__)




if __name__ == "__main__":
    main()
