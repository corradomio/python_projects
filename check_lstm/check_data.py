import pandas as pd
import pandasx as pdx
import torch as tc
import torchx as tcx


data = pdx.read_data("dfri.csv", ignore=['imp_date'], datetime=('imp_date', '%Y-%m', 'M'), index=['imp_date'])
data.info()

X, y = pdx.xy_split(data, target='import_kg')

X = X.to_numpy()
y = y.to_numpy().reshape((-1, 1))

Xt, yt = tcx.compose_data(y=y, X=X, slots=12, current=True, last=True)

dp = tcx.prepare_data(12, y=y, X=X, Xp=X, slots=12, current=True, last=True)
yp = dp.yp
for i in range(12):
    Xt = dp.compose(i)
    pass


pass
