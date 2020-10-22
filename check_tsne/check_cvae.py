import warnings
import silence_tensorflow as stf

warnings.simplefilter(action='ignore', category=FutureWarning)
stf.silence_tensorflow()


import matplotlib.pyplot as plt
import numpy as np
from cvae import cvae   # Initialise the tool, assuming we already have an array X containing the data
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data
y = np.array(list(map(int, mnist.target)))

embedder = cvae.CompressionVAE(X)   # Train the model
embedder.train()

X_embedding = embedder.embed(X)
print(X_embedding.shape)

# embedder.visualize(z, labels=[int(label) for label in mnist.target])

for i in range(10):
    Xe = X_embedding[y == i]
    plt.scatter(Xe[:, 0], Xe[:, 1], s=1)
plt.title("CVAE")
plt.show()
