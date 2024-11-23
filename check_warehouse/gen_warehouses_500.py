import numpy as np
from stdlib.jsonx import load, dump
from stdlib.tprint import tprint
from random import choice, uniform, choices
import matplotlib.pyplot as plt


def plot_locations_warehouses(n: int, locations: list[dict], warehouses: list[dict]):
    locs = []
    for l in locations:
        locs.append([l['lon'], l['lat']])
    locs = np.array(locs)

    warehouses = warehouses[:n]

    ware = []
    for w in warehouses:
        ware.append([w['lon'], w['lat']])
    ware = np.array(ware)

    plt.clf()
    plt.scatter(locs[:,0], locs[:, 1], c='blue', s=0.33, marker='.')
    plt.scatter(ware[:,0], ware[:, 1], c='red', s=1)

    plt.gca().set_aspect(1.5)
    plt.tight_layout()
    plt.savefig(f"wh_{n:03}.png", dpi=600)
# end


def gen_warehouses(N: int, locations):
    warehouse_dict: dict[str, dict] = {}
    for i in range(N):
        l1 = choice(locations)
        # {
        #   'name': str
        #   'lon': double
        #   'lat': double
        # }
        name = f"W{i+1:03}"
        lon = l1['lon'] + uniform(-0.1, 0.1)
        lat = l1['lat'] + uniform(-0.1, 0.1)

        warehouse_dict[name] = {
            "name": name,
            "lon": lon,
            "lat": lat
        }
    # end


    data = {
        "warehouses": warehouse_dict
    }

    dump(data, "data/warehouses_500.json")
    return warehouse_dict
# end


def main():
    data = load("data/warehouses_locations_uk.json")
    location_dict: dict[str, dict] = data["locations"]
    locations: list[dict] = list(location_dict.values())
    tprint(f"Locations: {len(locations)}")

    N = 500
    warehouse_dict = gen_warehouses(N, locations)
    warehouses = list(warehouse_dict.values())

    plot_locations_warehouses( 50, locations, warehouses)
    plot_locations_warehouses(100, locations, warehouses)
    plot_locations_warehouses(200, locations, warehouses)
    plot_locations_warehouses(500, locations, warehouses)

    pass


if __name__ == "__main__":
    main()
