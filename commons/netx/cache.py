from collections import defaultdict
from .graph import Graph

#
# It is used for netx.Graph AND nx.Graph
# Note: the caches are not automatically deleted on adding/removing nodes/edges
#  It is necessary to clear them 'explicitly'.
#

def check_cache(G: Graph):
    if '__netx_cache__' not in G.__dict__:
        G.__dict__["__netx_cache__"] = defaultdict(lambda : dict())


def clear_caches(G: Graph):
    if '__netx_cache__' in G.__dict__:
        G.__netx_cache__.clear()
    if '__networkx_cache__' in G.__dict__:
        G.__networkx_cache__.clear()
