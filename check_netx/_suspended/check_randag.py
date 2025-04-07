import matplotlib.pyplot as plt

import netx


def main():
    G = netx.random_dag(20, 19)
    G.dump()
    H = G.clone("H")
    H.dump()

    plt.gca()
    netx.draw(G)
    plt.title(G.name)
    plt.show()
    print(G)

    plt.gca()
    netx.draw(H)
    plt.title(H.name)
    plt.show()
    print(H)



if __name__ == '__main__':
    main()
