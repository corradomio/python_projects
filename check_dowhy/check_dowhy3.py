import networkx as nx
import matplotlib.pyplot as plt
from dowhy import gcm
import numpy as np, pandas as pd
from pprint import pprint

causal_graph = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])
nx.draw_networkx(causal_graph, with_labels=True)
plt.show()
causal_model = gcm.StructuralCausalModel(causal_graph)

X = np.random.normal(loc=0, scale=1, size=1000)
Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))

pprint(data.head())

gcm.auto.assign_causal_mechanisms(causal_model, data)

gcm.fit(causal_model, data)
print('---')
samples = gcm.interventional_samples(causal_model, {'Y': lambda y: 2.34}, num_samples_to_draw=1000)
pprint(samples.head())
print('---')
samples = gcm.interventional_samples(causal_model, {'Y': lambda y: 1.5}, num_samples_to_draw=1000)
pprint(samples.head())

