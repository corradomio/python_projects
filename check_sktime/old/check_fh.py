from sktime.forecasting.base import ForecastingHorizon


def main():
    fh = ForecastingHorizon(10)
    fh = ForecastingHorizon([10])
    fh = ForecastingHorizon([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pass


if __name__ == "__main__":
    main()

