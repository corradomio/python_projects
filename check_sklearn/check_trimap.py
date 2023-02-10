import matplotlib.pyplot as plt
import trimap
import umap
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("load digits")
digits = load_digits()

print("trimap digits")
embedding = trimap.TRIMAP().fit_transform(digits.data)
selected = np.random.choice(range(len(embedding)), 1000)
emb = embedding[selected, :]
plt.scatter(emb[:, 0], emb[:, 1])
plt.show()

print("umap digits")
embedding = umap.UMAP().fit_transform(digits.data)
selected = np.random.choice(range(len(embedding)), 1000)
emb = embedding[selected, :]
plt.scatter(emb[:, 0], emb[:, 1])
plt.show()

print("end")
pass
