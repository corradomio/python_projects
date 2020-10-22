import umap
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data
y = np.array(list(map(int, mnist.target)))

# X = X[0:100]
# y = y[0:100]

X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP()
X_embedding = reducer.fit_transform(X_scaled)
print(X_embedding.shape)

for i in range(10):
    Xe = X_embedding[y == i]
    plt.scatter(Xe[:, 0], Xe[:, 1], s=1)
plt.title("UMAP")
plt.show()
