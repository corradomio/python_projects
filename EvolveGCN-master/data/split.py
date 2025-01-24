import csv


def main():
    graphs = [[] for i in range(50)]
    nodes = [set() for i in range(50)]
    graph = -1
    with open("sbm_50t_1000n_adj.csv", mode="r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=",")
        next(r)
        for e in r:
            # source, targer, weight, g
            s,t,w,g = list(map(int, e))

            if graph != g:
                print(f"graph {g}")
                graph = g

            graphs[graph].append((s,t))
            nodes[graph].add(s)
            nodes[graph].add(t)
        # end
    # end

    for i in range(50):
        n_list = sorted(nodes[i])
        n_nodes = len(n_list)

        with open(f"sbm_50t_1000n_adj_{i}.csv", mode="w", encoding="utf-8") as f:
            f.writelines("source,target\n")
            for e in graphs[i]:
                u = e[0]
                v = e[1]
                f.write(f"{u},{v}\n")
            # end
        # end
    # end
    pass
# end




if __name__ == "__main__":
    main()
