import matplotlib.pyplot as plt
import numpy as np

from stdlib.jsonx import load
from stdlib.tprint import tprint

def plot(N):
    # data = load("data/warehouses_locations_uk.json")
    # warehouses = data["warehouses"]
    # locations = data["locations"]

    # N = 50

    locations = load("data/locations_uk.json")["locations"]
    warehouses = load("data/warehouses_500.json")["warehouses"]

    n = len(warehouses)
    m = len(locations)

    tprint(f"warehouses: {n}")
    tprint(f" locations: {m}")

    locs = np.zeros((m, 2), dtype=float)
    for j, lj in enumerate(locations):
        coords = locations[lj]
        locs[j,0] = coords["lon"]
        locs[j,1] = coords["lat"]

    ware = np.zeros((N, 2), dtype=float)
    for i, wi in enumerate(warehouses):
        if i >= N: break
        coords = warehouses[wi]
        ware[i, 0] = coords["lon"]
        ware[i, 1] = coords["lat"]

    plt.clf()
    plt.figure(figsize=(3, 5))
    plt.scatter(locs[:,0], locs[:, 1], c='green', s=0.20, marker='.')
    plt.scatter(ware[:,0], ware[:, 1], c='red', s=2)

    plt.gca().set_aspect(1.5)
    plt.tight_layout()
    plt.savefig(f'wh_{N:03}.png', dpi=600)

def main():
    plot(50)
    plot(60)
    plot(80)
    plot(100)
    plot(200)
    plot(500)

# end


if __name__ == "__main__":
    main()
