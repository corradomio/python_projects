import matplotlib.pyplot as plt
import networkx as nx

import netx
import stdlib.logging as logging
from castlex.datasets import simulator as simx


def main():
    G = netx.random_dag(6, 7, create_using=nx.DiGraph)
    netx.draw(G)
    plt.show()

    # W = netx.adjacency_matrix(G)

    iidsim = simx.IIDSimulation()
    iidsim.fit(G)

    data = iidsim.generate(n=10000)
    pass



if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
