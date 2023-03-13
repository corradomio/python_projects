import igraph as ig
from igraph import *

print(ig.__version__)

g = Graph(edges=[(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)], directed=True)

print(g)

g = Graph()
g.add_vertices(3)
g.add_edges([(0,1), (1,2)])

print(g)