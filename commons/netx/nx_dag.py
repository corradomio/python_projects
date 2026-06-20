from typing import Iterable

import networkx as nx

def transitive_reduction(G: nx.DiGraph, node_attrs:bool=False, edge_attrs:bool=False):
    # extends nx.transitive_reduction to transfer
    # node and edge attributes to the new graph
    T = nx.transitive_reduction(G)

    if node_attrs:
        for t in T.nodes:
            T.nodes[t].update(G.nodes[t])

    if edge_attrs:
        for e in T.edges:
            T.edges[e].update(G.edges[e])

    return T


def find_sources(G: nx.DiGraph) -> list:
    sources = []
    for n in G:
        if G.in_degree(n) == 0:
            sources.append(n)
    return sources


def find_sinks(G: nx.DiGraph, node_set: Iterable|set|list) -> list:
    sinks = []
    for n in node_set:
        if G.out_degree(n) == 0:
            sinks.append(n)
    return sinks



def remove_duplicate_clusters(clusters: dict[str, set[str]]):
    # remove duplicated
    cluster_ids = list(clusters.keys())
    n_clusters = len(clusters)
    to_remove = set()
    for i in range(n_clusters):
        icid = cluster_ids[i]
        icluster = clusters[icid]
        if len(icluster) == 0: continue
        for j in range(i+1,n_clusters):
            jcid = cluster_ids[j]
            jcluster = clusters[jcid]
            if len(jcluster) == 0: continue
            if icluster == jcluster:
                to_remove.add(jcid)
    # end
    for cid in to_remove:
        if cid in clusters:
            del clusters[cid]

    return clusters


def transitive_clusters(G: nx.DiGraph) -> dict:
    clusters = {}
    nodes = [n for n in G if G.in_degree[n] == 0]
    processed = set()
    for n in nodes:
        if n in processed: continue

        cluster: set = nx.descendants(G, n)
        processed.add(n)
        processed = processed.union(cluster)
        clusters[n] = cluster
    # end

    assert len(processed) == G.number_of_nodes()

    return clusters
