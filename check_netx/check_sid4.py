from pprint import pprint
import netx
import numpy as np

G = np.array((
   (0,1,1,0,0,0,0,0,0,0),
   (0,0,1,1,0,0,1,1,0,1),
   (0,0,0,0,0,1,1,1,0,1),
   (0,0,0,0,1,1,1,1,1,1),
   (0,0,0,0,0,1,0,1,0,0),
   (0,0,0,0,0,0,1,0,0,1),
   (0,0,0,0,0,0,0,1,1,1),
   (0,0,0,0,0,0,0,0,0,1),
   (0,0,0,0,0,0,0,0,0,0),
   (0,0,0,0,0,0,0,0,0,0)
))
H = np.array((
   (0,0,0,0,0,0,0,0,0,0),
   (1,0,0,0,0,0,0,0,0,0),
   (1,1,0,0,0,0,0,0,0,0),
   (0,1,0,0,1,0,0,0,0,0),
   (0,0,0,1,0,0,0,0,0,0),
   (0,0,1,0,0,0,0,0,0,0),
   (0,0,0,0,0,1,0,0,1,0),
   (0,1,0,0,0,0,0,0,1,0),
   (0,1,0,0,0,1,1,1,0,0),
   (0,0,1,0,0,0,1,1,1,0)
))

print(netx.metrics.structural_intervention_distance(G, H))

for Hi in netx.enumerate_all_directed_adjacency_matrices(H):
   print(netx.metrics.structural_intervention_distance(G, Hi))
   # pprint(Hi)

print(netx.metrics.structural_intervention_distance(G, H))

