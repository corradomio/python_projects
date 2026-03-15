import numpy as np

from sidimpl.structIntervDist import struct_interv_dist as structIntervDist

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

# print(structIntervDist(am(G), am(H1)))
print(structIntervDist(G, H2))
# print(structIntervDist(am(G), am(H2)))
