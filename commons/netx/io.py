import networkx as nx


def read_vecsv(path: str, comments="#", header=True, separator=",", create_using=None, direct=True) -> nx.Graph:
    # -vertices.csv
    # -edges.csv
    def vecsv_strip(path: str) -> str:
        suffix = "-vertices.csv"
        if path.endswith(suffix):
            return path[0:-len(suffix)]
        suffix = "-edges.csv"
        if path.endswith(suffix):
            return path[0:-len(suffix)]
        suffix = ".csv"
        if path.endswith(suffix):
            return path[0:-len(suffix)]
        else:
            return path
    # end

    def vecsv_vertices_file(path: str) -> str:
        return vecsv_strip(path) + "-vertices.csv"

    def vecsv_edges_file(path: str) -> str:
        return vecsv_strip(path) + "-edges.csv"

    def parse(s):
        try:
            return int(s)
        except:
            pass
        try:
            return float(s)
        except:
            pass
        return s
    # end

    vfile = vecsv_vertices_file(path)
    efile = vecsv_edges_file(path)

    if create_using is not None:
        g = create_using
    elif direct:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    # read vertices
    columns = None
    with open(vfile) as vfin:
        for line in vfin:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(comments):
                continue
            if columns is None:
                if header:
                    columns = line.split(separator)
                    continue
                else:
                    ncols = len(line.split(separator))
                    columns = [f"c{i+1:02}" for i in range(ncols)]
            # end
            props = list(map(parse, line.split(separator)))
            node = props[0]

            g.add_node(node)
            nattrs = dict()
            for i in range(1, len(columns)):
                nattrs[columns[i]] = props[i]
            g.nodes_[node].update(nattrs)
        # end

    # read edges
    columns = None
    with open(efile) as efin:
        for line in efin:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(comments):
                continue
            if columns is None:
                if header:
                    columns = line.split(separator)
                    continue
                else:
                    ncols = len(line.split(separator))
                    columns = [f"c{i + 1:02}" for i in range(ncols)]
            # end
            edge = list(map(parse, line.split(separator)))
            source = edge[0]
            target = edge[1]
            g.add_edge(source, target)
        # end
    return g
# end
