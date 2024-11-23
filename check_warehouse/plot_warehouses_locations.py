import matplotlib.pyplot as plt
import numpy as np

from stdlib.jsonx import load
from stdlib.tprint import tprint


def main():
    data = load("data/warehouses_locations_uk.json")
    warehouses = data["warehouses"]
    locations = data["locations"]
    n = len(warehouses)
    m = len(locations)

    tprint(f"warehouses: {n}")
    tprint(f" locations: {m}")

    locs = np.zeros((m, 2), dtype=float)
    for j, lj in enumerate(locations):
        coords = locations[lj]
        locs[j,0] = coords["lon"]
        locs[j,1] = coords["lat"]

    ware = np.zeros((n, 2), dtype=float)
    for i, wi in enumerate(warehouses):
        coords = warehouses[wi]
        ware[i, 0] = coords["lon"]
        ware[i, 1] = coords["lat"]

    plt.clf()
    plt.scatter(locs[:,0], locs[:, 1], c='blue', s=0.33, marker='.')
    plt.scatter(ware[:,0], ware[:, 1], c='red', s=1)

    plt.gca().set_aspect(1.5)
    plt.tight_layout()
    plt.savefig('uk.png', dpi=600)
# end


if __name__ == "__main__":
    main()
