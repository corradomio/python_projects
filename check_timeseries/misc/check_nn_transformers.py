import numpy as np
import numpyx as npx


X = np.array([[r*100 + c for c in range(1, 5)] for r in range(1, 10)])
y = np.array([[1000+r] for r in range(1, 10)])

# ---------------------------------------------------------------------------
tt = npx.RNNTrainTransform(steps=2)

Xr, yr = tt.fit_transform(X, y)

Xt = np.transpose(Xr, axes=(0, 2, 1))


# ---------------------------------------------------------------------------
tt = npx.CNNTrainTransform(steps=2)

Xc, yc = tt.fit_transform(X, y)

# ---------------------------------------------------------------------------
pt = npx.CNNPredictTransform(steps=2)
n = 3
Xp = np.array([[600 + r*100 + c for c in range(1, 5)] for r in range(1, 1+n)])


yp = pt.fit(X, y).transform(Xp)
for i in range(n):
    xt = pt.step(i)
    yp[i, -1] = i+1
    pass

pass