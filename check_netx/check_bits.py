import netx
import networkx as nx
import numpy as np
from pprint import pprint
from typing import Iterator
from netx import power_adjacency_matrix, random_adjacency_matrix


# def _bitsgen(nbits: int) -> Iterator[list[int]]:
#     bits = [0]*nbits
#     ones = [1]*nbits
#     while bits != ones:
#         yield bits
#         for i in range(nbits):
#             if bits[i] == 0:
#                 bits[i] = 1
#                 break
#             else:
#                 bits[i] = 0
#     yield ones
# # end
#
#
# for bits in _bitsgen(4):
#     print(bits)


# def ilog2(x):
#     i, e = 0, 1
#     while e < x:
#         e *= 2
#         i += 1
#     return i
#
# for i in range(17):
#     print(i, ":", ilog2(i))

A = random_adjacency_matrix(6, 8, directed=True, loop=False, acyclic=False)
pprint(A)

G = netx.from_numpy_array(A)
netx.draw(G)
netx.show()

P = power_adjacency_matrix(A, -1)
pprint(P)

G = netx.from_numpy_array(P)
netx.draw(G)
netx.show()
