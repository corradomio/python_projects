from pprint import pprint
import numpy as np, pandas as pd, networkx as nx
from dowhy import gcm
np.random.seed(10)  # to reproduce these results

Z = np.random.normal(loc=0, scale=1, size=1000)
X = 2*Z + np.random.normal(loc=0, scale=1, size=1000)
Y = 3*X + 4*Z + np.random.normal(loc=0, scale=1, size=1000)
data = pd.DataFrame(dict(X=X, Y=Y, Z=Z))

causal_model = gcm.ProbabilisticCausalModel(nx.DiGraph([('Z', 'Y'), ('Z', 'X'), ('X', 'Y')]))
causal_model.set_causal_mechanism('Z', gcm.EmpiricalDistribution())
causal_model.set_causal_mechanism('X', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
gcm.fit(causal_model, data)

strength = gcm.arrow_strength(causal_model, 'Y')
pprint(strength)


def mean_diff(Y_old, Y_new):
    return np.mean(Y_new) - np.mean(Y_old)


pprint(gcm.arrow_strength(causal_model, 'Y', difference_estimation_func=mean_diff))

