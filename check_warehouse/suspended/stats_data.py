import matplotlib.pyplot as plt
from common import Data


def main():
    data = Data().load("_db")
    # data = Data().load()
    p = data.parts[0]

    w_coords = data.coordinates(locations=False)
    l_coords = data.coordinates(locations=True)

    plt.scatter(l_coords[:,0], l_coords[:,1], c='blue', s=.1)
    plt.scatter(w_coords[:,0], w_coords[:,1], c='red', s=5.)

    plt.gca().set_aspect(1)
    plt.show()

    N = data.neighborhoods()

    W_ = data.random_warehouses(10)
    D = data.distances(dst=W_, locations=False)
    C = data.weights(p, W_)
    S = data.in_stock(p).sum()
    S_ = (C*S).round().astype(dtype=int)
    print(S, S_.sum())
    pass


if __name__ == "__main__":
    main()
