import numpy as np
from stdlib.jsonx import load


def main():
    data = load("warehouses_locations_uk.json")
    warehouses = list(data["warehouses"].keys())
    locations = list(data["locations"].keys())
    distances = data["distances"]

    n = len(warehouses)
    m = len(locations)

    # widx = {w:wi for w, wi in enumerate(warehouses)}
    # lidx = {l:lj for l, lj in enumerate(locations)}

    A = np.zeros((n+m, n+m), dtype=np.int8)
    D = np.zeros((n+m, n+m), dtype=np.float32)
    D[:,:] = -1

    for i in range(n):
        wi = warehouses[i]
        for j in range(m):
            lj = locations[j]

            if wi not in distances:
                continue
            if lj not in distances[wi]:
                continue

            dij = distances[wi][lj]["dist_min"]

            A[i,n+j] = 1
            D[i,n+j] = dij
    # end

    np.savetxt("graph_adjacency_matrix.csv", A, fmt="%d", delimiter=",")
    np.savetxt("graph_distances.csv", D, fmt="%f", delimiter=",")
# end


if __name__ == "__main__":
    main()
