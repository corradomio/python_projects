
import gzip
import pickle

with open("D:/Dropbox/Datasets/mnist/mnist/mnist.pkl", mode='rb') as f:
    pickle.load(f, encoding="latin-1")

with gzip.open("D:/Dropbox/Datasets/mnist/mnist/mnist.pkl.gz", mode='rb') as f:
    ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")
