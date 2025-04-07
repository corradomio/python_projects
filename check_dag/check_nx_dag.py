import networkx as nx
import netx

def main():
    G = netx.random_dag(10, 20, connected=True, create_using=netx.DiGraph)

    for n in G.nodes():
        print(n)

    for e in G.edges():
        print(e)

    print(netx.sources(G))
    print(netx.destinations(G))

    print(netx.predecessors(G, 5))
    print(netx.successors(G, 5))
    pass



if __name__ == "__main__":
    main()
