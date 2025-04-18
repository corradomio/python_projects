from pprint import pprint
import networkx as nx
causal_graph = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])

from dowhy import gcm
causal_model = gcm.StructuralCausalModel(causal_graph)

import numpy as np, pandas as pd

X = np.random.normal(loc=0, scale=1, size=1000)
Y = 2 * X + np.random.normal(loc=0, scale=1, size=1000)
Z = 3 * Y + np.random.normal(loc=0, scale=1, size=1000)
data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))
pprint(data.head())

gcm.auto.assign_causal_mechanisms(causal_model, data)

gcm.fit(causal_model, data)

samples = gcm.interventional_samples(causal_model,
                                     {'Y': lambda y: 2.34 },
                                     num_samples_to_draw=1000)
pprint(samples.head())

causal_model = gcm.StructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')])) # X → Y → Z
strength_median, strength_intervals = gcm.confidence_intervals(
    gcm.bootstrap_training_and_sampling(gcm.direct_arrow_strength,
                                        causal_model,
                                        bootstrap_training_data=data,
                                        target_node='Y'))
pprint((strength_median, strength_intervals))
