import warnings
import silence_tensorflow as stf

warnings.simplefilter(action='ignore', category=FutureWarning)
stf.silence_tensorflow()

# Import CVAE
from cvae import cvae   # Initialise the tool, assuming we already have an array X containing the data
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data

embedder = cvae.CompressionVAE(X)   # Train the model
embedder.train()

z = embedder.embed(X)
embedder.visualize(z, labels=[int(label) for label in mnist.target])
