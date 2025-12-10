from tsai.basics import *

ts = get_forecasting_time_series("Sunspots").values
X, y = SlidingWindow(60, horizon=1)(ts)
splits = TimeSplitter(235)(y)
tfms = [None, TSForecasting()]
batch_tfms = TSStandardize()
fcst = TSForecaster(
    X, y, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=512, arch="TSTPlus", metrics=mae
    # , cbs=ShowGraph()
)
fcst.fit_one_cycle(50, 1e-3)
fcst.export("fcst.pkl")
