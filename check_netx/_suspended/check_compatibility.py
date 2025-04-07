import logging.config
import networkx as nx
import netx as nxx


def main():

    g: nxx.Graph = nxx.random_dag(20, 100)
    print(g)

    for n in g:
        print(n)

    g.has_node(1)
    pass


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
