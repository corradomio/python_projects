from category_encoders import *
import pandas as pd
from sklearn.datasets import load_boston
from pprint import pprint

bunch = load_boston()
y = bunch.target
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
# enc = BinaryEncoder(cols=['CHAS', 'RAD']).fit(X, y)
# enc = OneHotEncoder(cols=['CHAS', 'RAD']).fit(X, y)
# enc = CountEncoder(cols=['CHAS', 'RAD']).fit(X, y)
# enc = HashingEncoder(cols=['CHAS', 'RAD'], max_process=1).fit(X, y)
# enc = HelmertEncoder(cols=['CHAS', 'RAD']).fit(X, y)
enc = JamesSteinEncoder(cols=['CHAS', 'RAD']).fit(X, y)
numeric_dataset = enc.transform(X)
pprint(numeric_dataset.head())
