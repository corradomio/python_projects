import matplotlib.pyplot as plt
from stdlib.jsonx import load


ITEMS = [
    "700001",
    "700002",
    "700003",
    "700004",
    "700005",
    "700006",
    "700007",
    "700008",
    "700009",
    "700010",
]


def plot_locations(locations: dict):
    # locations = load("data_synth/new/locations_uk.json")["locations"]
    print(len(locations))

    plt.clf()
    llon = []
    llat = []
    for l in locations.values():
        lon = l["lon"]
        lat = l["lat"]

        llon.append(lon)
        llat.append(lat)
    # end
    plt.scatter(llon, llat, c='blue', s=0.33, marker='.')
    plt.gca().set_aspect(1.5)
    plt.gca().set_axis_off()
    plt.gcf().set_size_inches(4, 6)
    plt.tight_layout()

    plt.savefig("plots/uk.png", dpi=300)
    # plt.show()



def plot_warehouses(nw: int, locations: dict, warehouses: dict, stock_star: dict, item: str):
    print(f"{nw}-{item}")

    plt.clf()

    # plot locations
    llon = []
    llat = []
    for l in locations.values():
        lon = l["lon"]
        lat = l["lat"]

        llon.append(lon)
        llat.append(lat)
    plt.scatter(llon, llat, c='lightblue', s=0.33, marker='.')

    # plot warehouses
    llon_avl = []
    llat_avl = []
    llon_req = []
    llat_req = []

    for w in warehouses:
        if item in stock_star[w]:
            ss = stock_star[w][item]
        else:
            continue

        local_in_stock = ss["local_in_stock"]
        current_in_stock = ss["current_in_stock"]

        l = warehouses[w]
        lon = l["lon"]
        lat = l["lat"]

        llon.append(lon)
        llat.append(lat)

        if current_in_stock < local_in_stock:
            llon_req.append(lon)
            llat_req.append(lat)
        else:
            llon_avl.append(lon)
            llat_avl.append(lat)

        # if len(llon) == nw:
        #     break

    plt.scatter(llon_req, llat_req, c='red', s=3)
    plt.scatter(llon_avl, llat_avl, c='green', s=3)

    plt.gca().set_aspect(1.5)
    plt.gca().set_axis_off()
    plt.gcf().set_size_inches(4, 6)
    plt.tight_layout()

    plt.savefig(f"plots/{nw}/wh_{nw:03}-{item}.png", dpi=300)
# end


def select_warehouses(nw: int, warehouses: dict) -> dict:
    selected = {}
    for w in warehouses:
        if int(w[1:]) < nw:
            selected[w] = warehouses[w]
    return selected
# end


def main():
    locations = load("data_synth/new/locations_uk.json")["locations"]
    all_warehouses = load("data_synth/new/warehouses_500.json")["warehouses"]

    # plot_locations(locations)
    # for nw in [100,200, 500]:
    for nw in [50,60,70,80,90,100,200,500]:
        warehouses = select_warehouses(nw, all_warehouses)
        stock_star = load(f"data_synth/new/stock_star_{nw}.json")

        for item in ITEMS:
            plot_warehouses(nw, locations, warehouses, stock_star, item)
    pass


if __name__ == "__main__":
    main()

