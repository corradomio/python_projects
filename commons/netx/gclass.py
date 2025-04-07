import networkx.classes as nxc

from .graph import Graph


def is_netx_graph(G):
    gclass = G.__class__
    return gclass in {Graph}


def is_networkx_graph(G):
    gclass = G.__class__
    return gclass in {nxc.Graph, nxc.DiGraph, nxc.MultiGraph, nxc.MultiDiGraph}
