import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data
y = np.array(list(map(int, mnist.target)))

# X = X[0:100]
# y = y[0:100]

X_embedded = TSNE(n_components=2, n_jobs=6).fit_transform(X)
print(X_embedded.shape)

for i in range(10):
    Xe = X_embedded[y == i]
    plt.scatter(Xe[:, 0], Xe[:, 1], s=1)
plt.title("tSNE")
plt.show()
