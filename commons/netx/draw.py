import numpy as np
import networkx as nx
from .daggen import is_symmetric


def draw(G: np.ndarray | nx.Graph, pos=None, ax=None, labels=True, **kwds):
    """
    Little extended version of nx.draw()

    :param G: graph to draw. It can be a adjacency matrix, or a ``networkx.Graph'' or a
              local Graph object
    :param pos: positions of the nodes (optional)
    :param ax: matplotlib axis to use
    :param labels: if to visualize the nodes labels
    :param kwds: other parameters passed to ``networkx.draw()''
    :return:
    """
    if isinstance(G, np.ndarray):
        M = G
        if is_symmetric(M):
            G = nx.Graph(M)
        else:
            G = nx.DiGraph(M)
    if pos is None:
        pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, ax=ax, **kwds)
    if labels:
        nx.draw_networkx_labels(G, pos=pos)
# end


def show():
    import matplotlib.pyplot as plt
    plt.show()
# end
