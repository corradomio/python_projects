from collections import defaultdict
from .graph import Graph


def check_cache(G: Graph):
    if 'cache' not in G.__dict__:
        G.__dict__["cache"] = defaultdict(lambda : {})

