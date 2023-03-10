import pandas as pd

def to_gml(G):
    def node_of(n):
        return f"'{n}'" if ' ' in n else n

    gml = "digraph {\n"
    for n in G.nodes:
        gml += f"  {node_of(n)};\n"
    for e in G.edges:
        source = node_of(e[0])
        target = node_of(e[1])
        gml += f"  {source}->{target};\n"
    # end
    gml += "}"
    return gml


def remove_column_spaces(df: pd.DataFrame) -> pd.DataFrame:
    to_remove = []
    for col in df.columns:
        if ' ' in col:
            ncol = col.replace(' ', '_')
            df[ncol] = df[col]
            to_remove.append(col)
    # end
    df = df[df.columns.difference(to_remove)]
    return df
# end


def edges(*edges):
    elist = []
    def parse_edges(e):
        if isinstance(e[0], str) and isinstance(e[1], str):
            # elist.append((e[0], e[1]))
            source = [e[0]]
            target = [e[1]]
        elif isinstance(e[0], (list, tuple)) and isinstance(e[1], str):
            source = e[0]
            target = [e[1]]
        elif isinstance(e[0], str) and isinstance(e[1], (list, tuple)):
            source = [e[0]]
            target = e[1]
        else:
            source = e[0]
            target = e[1]
        for s in source:
            for t in target:
                elist.append((s, t))
                # yield (s, t)

    for e in edges:
        if len(e) > 2:
            for i in range(len(e)-1):
                parse_edges(e[i:i+2])
        else:
            parse_edges(e)

    return elist
    # raise StopIteration()
# end
