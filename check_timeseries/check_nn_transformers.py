import numpy as np
import numpyx as npx


X = np.array([[r*100 + c for c in range(1, 5)] for r in range(1, 10)])
y = np.array([[900+r] for r in range(1, 10)])

tt = npx.CNNTrainTransform(steps=2)

Xt, yt = tt.fit_transform(X, y)

# ---------------------------------------------------------------------------
pt = npx.CNNPredictTransform(steps=2)
n = 3
Xp = np.array([[r*600 + c for c in range(1, 5)] for r in range(1, 1+n)])
yp = pt.fit(X, y).transform(Xp)

for i in range(n):
    xt = pt.step(i)
    pass

pass