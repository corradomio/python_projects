import stdlib.logging as logging
import stdlib.csvx as csvx
import networkx as nx
import netx
import matplotlib.pyplot as plt


def add_iid(G, iid: str):

    n = "start"

    for c in iid:
        G.add_edge(n, c)
        n = c

    G.add_edge(n, "end")
# end


def main():
    invoices = csvx.load_csv("Data_sample_Masked.csv", skiprows=1, dtype=[None, str, None] + [str, str, str])

    invoice_graphs = {}

    cnt = 0
    for invoices in invoices:
        type = invoices[0]
        name = invoices[3] if type == "BUYER" else invoices[2]

        if name not in invoice_graphs:
            G = netx.Graph(direct=True)
            netx.add_nodes(G, ["start", "end"])
            invoice_graphs[name] = G
        else:
            G = invoice_graphs[name]

        add_iid(G, invoices[1])
        # cnt += 1
        # if cnt == 10:
        #     break

    for name in invoice_graphs:
        G = invoice_graphs[name]
        plt.title(name)
        netx.draw(G)
        plt.savefig(f"plots/{name}.png", dpi=300)
        plt.close()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

