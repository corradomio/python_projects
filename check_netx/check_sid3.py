import numpy as np
import netx


G = np.array(
    [[0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]])


H1 = np.array(
    [[0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]])


H2 = np.array(
    [[0, 0, 1, 1, 1],
     [1, 0, 1, 1, 1],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]])


print(netx.metrics.structural_intervention_distance(G, H1))
print(netx.metrics.structural_intervention_distance(G, H2))
print(netx.metrics.structural_intervention_distance(H1, H2))
