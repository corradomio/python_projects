from sktime.forecasting.naive import NaiveForecaster

X = None
y = None
fh = None

nf = NaiveForecaster()
nf.update(y, X, False)


