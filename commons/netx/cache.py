from collections import defaultdict
from .graph import Graph


def check_cache(G: Graph):
    if '__netx_cache__' not in G.__dict__:
        G.__dict__["__netx_cache__"] = defaultdict(lambda : dict())

